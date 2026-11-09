from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from shoppilot.data.gold.quality import (
    calculate_assignment_metrics,
    calculate_fact_metrics,
    calculate_signal_metrics,
    calculate_user_feature_metrics,
    validate_gold_v2_metrics,
)
from shoppilot.data.signatures import calculate_dataset_signature
from shoppilot.jobs.gold import (
    ASSIGNMENT_COLUMNS,
    EVENT_COLUMNS,
    OUTPUT_PARTITIONS,
    SIGNAL_COLUMNS,
    USER_FEATURE_COLUMNS,
    _prepare_for_write,
    _signature_specs,
)
from shoppilot.data.gold.transforms import (
    assign_global_calendar_roles,
    attach_current_catalog_status,
    build_rating_signals,
    build_user_features_at_cutoff,
    build_user_item_events,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("GoldV2Tests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def make_reviews(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        review_id string,
        product_id string,
        user_id string,
        rating double,
        verified_purchase boolean,
        helpful_vote long,
        event_time timestamp,
        source_record_id string
        """,
    )


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def make_signals(spark, rows):
    reviews = make_reviews(spark, rows)
    events = build_user_item_events(reviews, "snapshot-test")
    return events, build_rating_signals(events)


def test_user_item_events_keep_all_ratings_without_model_columns(spark):
    reviews = make_reviews(
        spark,
        [
            ("t", "r1", "p1", "u", 5.0, True, 7, utc(2020, 1, 1), "s1"),
            ("t", "r2", "p2", "u", 3.0, False, 2, utc(2020, 2, 1), "s2"),
            ("t", "r3", "p3", "u", 1.0, True, 0, utc(2020, 3, 1), "s3"),
        ],
    )

    events = build_user_item_events(reviews, "snapshot-test")
    rows = events.orderBy("event_id").collect()

    assert [row["event_id"] for row in rows] == ["r1", "r2", "r3"]
    assert [row["rating"] for row in rows] == [5.0, 3.0, 1.0]
    assert rows[0]["helpful_vote"] == 7
    assert rows[0]["event_type"] == "review_rating"
    assert rows[0]["snapshot_id"] == "snapshot-test"
    assert "preference" not in events.columns
    assert "confidence" not in events.columns
    assert "role" not in events.columns


def test_current_catalog_status_keeps_orphan_events(spark):
    events, _ = make_signals(
        spark,
        [
            ("t", "r1", "p-known", "u", 5.0, True, 0, utc(2020, 1, 1), "s1"),
            ("t", "r2", "p-orphan", "u", 4.0, True, 0, utc(2020, 2, 1), "s2"),
        ],
    )
    products = spark.createDataFrame(
        [("t", "p-known", True)],
        "tenant_id string, product_id string, is_active boolean",
    )

    actual = {
        row["product_id"]: row["is_missing_current_catalog"]
        for row in attach_current_catalog_status(events, products).collect()
    }

    assert actual == {"p-known": False, "p-orphan": True}


def test_rating_signal_mapping_is_uniform_and_helpful_vote_is_not_used(spark):
    reviews = make_reviews(
        spark,
        [
            ("t", "r5", "p5", "u", 5.0, True, 999, utc(2020, 1, 1), "s5"),
            ("t", "r4", "p4", "u", 4.0, True, 0, utc(2020, 2, 1), "s4"),
            ("t", "r3", "p3", "u", 3.0, True, 100, utc(2020, 3, 1), "s3"),
            ("t", "r1", "p1", "u", 1.0, True, 100, utc(2020, 4, 1), "s1"),
        ],
    )
    signals = build_rating_signals(
        build_user_item_events(reviews, "snapshot-test")
    )
    actual = {row["event_id"]: row.asDict() for row in signals.collect()}

    for event_id in ("r5", "r4"):
        assert actual[event_id]["feedback_class"] == "positive"
        assert actual[event_id]["preference"] == 1.0
        assert actual[event_id]["confidence"] == 1.0
        assert actual[event_id]["is_training_eligible"] is True

    assert actual["r3"]["feedback_class"] == "neutral"
    assert actual["r1"]["feedback_class"] == "negative"
    for event_id in ("r3", "r1"):
        assert actual[event_id]["preference"] is None
        assert actual[event_id]["confidence"] is None
        assert actual[event_id]["is_training_eligible"] is False
    assert "helpful_vote" not in signals.columns


def test_user_features_use_only_history_before_cutoff(spark):
    _, signals = make_signals(
        spark,
        [
            ("t", "r1", "p1", "u", 5.0, True, 0, utc(2020, 1, 1), "s1"),
            ("t", "r2", "p2", "u", 3.0, True, 0, utc(2020, 2, 1), "s2"),
            ("t", "r3", "p3", "u", 1.0, True, 0, utc(2020, 3, 1), "s3"),
            ("t", "future", "p4", "u", 5.0, True, 0, utc(2021, 1, 1), "s4"),
        ],
    )

    row = build_user_features_at_cutoff(
        signals,
        "2021-01-01",
    ).first()

    assert row["all_interaction_count"] == 3
    assert row["positive_interaction_count"] == 1
    assert row["neutral_interaction_count"] == 1
    assert row["negative_interaction_count"] == 1
    assert row["last_event_time"].date().isoformat() == "2020-03-01"
    assert row["feature_cutoff"].date().isoformat() == "2021-01-01"


def test_global_calendar_roles_use_shared_boundaries(spark):
    _, signals = make_signals(
        spark,
        [
            ("t", "train", "p1", "u", 5.0, True, 0, utc(2020, 12, 31, 23, 59), "s1"),
            ("t", "validation", "p2", "u", 5.0, True, 0, utc(2021, 1, 1), "s2"),
            ("t", "test", "p3", "u", 5.0, True, 0, utc(2022, 1, 1), "s3"),
            ("t", "oot", "p4", "u", 5.0, True, 0, utc(2023, 1, 1), "s4"),
        ],
    )

    actual = {
        row["event_id"]: row["role"]
        for row in assign_global_calendar_roles(signals).collect()
    }

    assert actual == {
        "train": "train",
        "validation": "validation",
        "test": "test",
        "oot": "oot_stress",
    }


def test_availability_state_is_computed_from_history_before_each_window(spark):
    _, signals = make_signals(
        spark,
        [
            ("t", "negative-history", "p0", "u-nonpositive", 1.0, True, 0, utc(2020, 1, 1), "s0"),
            ("t", "short-history", "p1", "u-short", 5.0, True, 0, utc(2020, 2, 1), "s1"),
            ("t", "warm-1", "p1", "u-warm", 5.0, True, 0, utc(2020, 3, 1), "s2"),
            ("t", "warm-2", "p2", "u-warm", 4.0, True, 0, utc(2020, 4, 1), "s3"),
            ("t", "target-new", "p-new", "u-new", 5.0, True, 0, utc(2021, 2, 1), "s4"),
            ("t", "target-nonpositive", "p-new", "u-nonpositive", 5.0, True, 0, utc(2021, 2, 1), "s5"),
            ("t", "target-short", "p1", "u-short", 5.0, True, 0, utc(2021, 2, 1), "s6"),
            ("t", "target-warm", "p2", "u-warm", 5.0, True, 0, utc(2021, 2, 1), "s7"),
        ],
    )
    rows = (
        assign_global_calendar_roles(signals)
        .filter(F.col("is_evaluation_target"))
        .collect()
    )
    actual = {row["event_id"]: row.asDict() for row in rows}

    assert actual["target-new"]["user_availability_state"] == "platform_cold"
    assert actual["target-nonpositive"]["user_availability_state"] == "als_cold_nonpositive_only"
    assert actual["target-short"]["user_availability_state"] == "short_positive_history"
    assert actual["target-warm"]["user_availability_state"] == "warm_positive_2_4"
    assert actual["target-new"]["item_availability_state"] == "unknown_to_als"
    assert actual["target-short"]["item_availability_state"] == "known_to_als"
    assert actual["target-warm"]["history_depth_before_cutoff"] == 2


def test_nonpositive_validation_event_is_not_an_evaluation_target(spark):
    _, signals = make_signals(
        spark,
        [
            ("t", "negative", "p1", "u", 1.0, True, 0, utc(2021, 2, 1), "s1"),
        ],
    )
    row = assign_global_calendar_roles(signals).first()

    assert row["role"] == "validation"
    assert row["is_evaluation_target"] is False
    assert row["user_availability_state"] is None
    assert row["item_availability_state"] is None


def test_same_user_timestamp_bucket_has_one_role_and_id(spark):
    event_time = utc(2022, 6, 1)
    _, signals = make_signals(
        spark,
        [
            ("t", "r1", "p1", "u", 5.0, True, 0, event_time, "s1"),
            ("t", "r2", "p2", "u", 4.0, True, 0, event_time, "s2"),
        ],
    )

    rows = assign_global_calendar_roles(signals).collect()

    assert {row["role"] for row in rows} == {"test"}
    assert len({row["target_bucket_id"] for row in rows}) == 1


def test_gold_v2_quality_metrics_accept_valid_contract(spark):
    reviews = make_reviews(
        spark,
        [
            ("t", "r1", "p1", "u", 5.0, True, 0, utc(2020, 1, 1), "s1"),
            ("t", "r2", "p2", "u", 3.0, True, 0, utc(2021, 1, 1), "s2"),
            ("t", "r3", "p1", "u", 4.0, True, 0, utc(2022, 1, 1), "s3"),
        ],
    )
    products = spark.createDataFrame(
        [("t", "p1", True), ("t", "p2", True)],
        "tenant_id string, product_id string, is_active boolean",
    )
    events = attach_current_catalog_status(
        build_user_item_events(reviews, "snapshot-test"),
        products,
    )
    signals = build_rating_signals(events)
    assignments = assign_global_calendar_roles(signals)
    features = build_user_features_at_cutoff(signals, "2022-01-01")

    metrics = {}
    metrics.update(calculate_fact_metrics(reviews, events))
    metrics.update(calculate_signal_metrics(events, signals))
    metrics.update(calculate_assignment_metrics(signals, assignments))
    metrics.update(calculate_user_feature_metrics(features))

    validate_gold_v2_metrics(metrics)
    assert metrics["review_event_parity_difference_count"] == 0
    assert metrics["event_signal_parity_difference_count"] == 0
    assert metrics["signal_assignment_parity_difference_count"] == 0
    assert metrics["future_event_feature_count"] == 0


def test_gold_v2_quality_gate_reports_contract_violation():
    with pytest.raises(ValueError, match="invalid_positive_signal_count: 1"):
        validate_gold_v2_metrics({"invalid_positive_signal_count": 1})


def test_write_boundary_coalesces_small_file_prone_outputs(spark):
    dataframe = spark.range(100).repartition(20)

    assignments = _prepare_for_write(
        "evaluation_assignments",
        dataframe,
    )
    features = _prepare_for_write("user_features", dataframe)

    assert assignments.rdd.getNumPartitions() == OUTPUT_PARTITIONS[
        "evaluation_assignments"
    ]
    assert features.rdd.getNumPartitions() == OUTPUT_PARTITIONS["user_features"]


def test_signature_specs_cover_every_published_contract_column():
    specs = _signature_specs()

    assert specs["user_item_events"][1] == list(EVENT_COLUMNS)
    assert specs["interaction_signals"][1] == list(SIGNAL_COLUMNS)
    assert specs["evaluation_assignments"][1] == list(ASSIGNMENT_COLUMNS)
    assert specs["user_features"][1] == list(USER_FEATURE_COLUMNS)
    assert "document_text" in specs["product_documents"][1]
    assert "price" in specs["product_documents"][1]
    assert "snapshot_id" in specs["product_documents"][1]


def test_dataset_signature_distinguishes_null_empty_and_value(spark):
    schema = "id string, optional_text string"
    variants = [
        spark.createDataFrame([("same-key", None)], schema),
        spark.createDataFrame([("same-key", "")], schema),
        spark.createDataFrame([("same-key", "value")], schema),
    ]

    checksums = {
        calculate_dataset_signature(
            dataframe,
            ["id"],
            ["id", "optional_text"],
        )["stable_checksum"]
        for dataframe in variants
    }

    assert len(checksums) == 3


def test_dataset_signature_rejects_incomplete_checksum_contract(spark):
    dataframe = spark.createDataFrame([("id-1",)], "id string")

    with pytest.raises(ValueError, match="missing columns: payload"):
        calculate_dataset_signature(
            dataframe,
            ["id"],
            ["id", "payload"],
        )


def test_dataset_signature_normalizes_map_key_order(spark):
    schema = "id string, details map<string,string>"
    left = spark.createDataFrame(
        [("same-key", {"b": "2", "a": "1"})],
        schema,
    )
    right = spark.createDataFrame(
        [("same-key", {"a": "1", "b": "2"})],
        schema,
    )

    left_signature = calculate_dataset_signature(
        left,
        ["id"],
        ["id", "details"],
    )
    right_signature = calculate_dataset_signature(
        right,
        ["id"],
        ["id", "details"],
    )

    assert left_signature == right_signature
