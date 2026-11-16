from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.model_input import (
    MAPPING_SCOPE_COLUMNS,
    aggregate_binary_pairs,
    attach_mapping_indices,
    build_candidate_universe,
    build_item_mapping,
    build_item_popularity_counts,
    build_seen_items,
    build_user_mapping,
    calculate_mapping_coverage_metrics,
    select_training_events,
    validate_mapping_coverage_metrics,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderModelInputTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def make_signals(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        event_id string,
        user_id string,
        product_id string,
        event_time timestamp,
        signal_version string,
        feedback_class string,
        preference double,
        confidence double,
        is_training_eligible boolean,
        snapshot_id string
        """,
    )


def make_assignments(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        event_id string,
        user_id string,
        product_id string,
        event_time timestamp,
        signal_version string,
        protocol_version string,
        role string,
        is_training_eligible boolean,
        snapshot_id string
        """,
    )


def test_select_training_events_keeps_only_requested_positive_train_rows(spark):
    signals = make_signals(
        spark,
        [
            ("t", "keep", "u1", "p1", utc(2020, 6, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
            ("t", "neutral", "u2", "p2", utc(2020, 6, 1), "signal-v1", "neutral", None, None, False, "snapshot-v1"),
            ("t", "validation", "u3", "p3", utc(2021, 6, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
            ("t", "other-signal", "u4", "p4", utc(2020, 6, 1), "signal-v2", "positive", 1.0, 1.0, True, "snapshot-v1"),
        ],
    )
    assignments = make_assignments(
        spark,
        [
            ("t", "keep", "u1", "p1", utc(2020, 6, 1), "signal-v1", "protocol-v1", "train", True, "snapshot-v1"),
            ("t", "neutral", "u2", "p2", utc(2020, 6, 1), "signal-v1", "protocol-v1", "train", False, "snapshot-v1"),
            ("t", "validation", "u3", "p3", utc(2021, 6, 1), "signal-v1", "protocol-v1", "validation", True, "snapshot-v1"),
            ("t", "other-signal", "u4", "p4", utc(2020, 6, 1), "signal-v2", "protocol-v1", "train", True, "snapshot-v1"),
        ],
    )

    rows = select_training_events(
        signals,
        assignments,
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    ).collect()

    assert [row["event_id"] for row in rows] == ["keep"]
    assert rows[0]["preference"] == 1.0
    assert rows[0]["confidence"] == 1.0


def test_select_training_events_treats_cutoff_as_exclusive(spark):
    boundary = utc(2021, 1, 1)
    signals = make_signals(
        spark,
        [("t", "boundary", "u", "p", boundary, "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1")],
    )
    assignments = make_assignments(
        spark,
        [("t", "boundary", "u", "p", boundary, "signal-v1", "protocol-v1", "train", True, "snapshot-v1")],
    )

    result = select_training_events(
        signals,
        assignments,
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )

    assert result.count() == 0


def test_select_training_events_joins_on_complete_event_identity(spark):
    event_time = utc(2020, 6, 1)
    signals = make_signals(
        spark,
        [("t", "same-id", "actual-user", "p", event_time, "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1")],
    )
    assignments = make_assignments(
        spark,
        [("t", "same-id", "different-user", "p", event_time, "signal-v1", "protocol-v1", "train", True, "snapshot-v1")],
    )

    result = select_training_events(
        signals,
        assignments,
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )

    assert result.count() == 0


def make_training_events(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        event_id string,
        user_id string,
        product_id string,
        event_time timestamp,
        preference double,
        confidence double,
        snapshot_id string,
        signal_version string,
        protocol_version string
        """,
    )


def test_binary_pair_aggregation_keeps_one_uniformly_weighted_pair(spark):
    events = make_training_events(
        spark,
        [
            ("t", "e1", "u1", "p1", utc(2019, 1, 1), 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e2", "u1", "p1", utc(2020, 1, 1), 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e3", "u1", "p2", utc(2020, 2, 1), 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )

    rows = {
        (row["tenant_id"], row["user_id"], row["product_id"]): row.asDict()
        for row in aggregate_binary_pairs(events).collect()
    }

    assert set(rows) == {("t", "u1", "p1"), ("t", "u1", "p2")}
    assert rows[("t", "u1", "p1")]["event_count"] == 2
    assert rows[("t", "u1", "p1")]["preference"] == 1.0
    assert rows[("t", "u1", "p1")]["confidence"] == 1.0
    assert rows[("t", "u1", "p1")]["aggregation_version"] == "binary_unique_pair_v1"


def test_binary_pair_aggregation_is_tenant_and_version_scoped(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("tenant-a", "e1", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-b", "e2", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e3", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v2", "signal-v1", "protocol-v1"),
        ],
    )

    actual = {
        (
            row["tenant_id"],
            row["snapshot_id"],
            row["signal_version"],
            row["protocol_version"],
        ): row["event_count"]
        for row in aggregate_binary_pairs(events).collect()
    }

    assert actual == {
        ("tenant-a", "snapshot-v1", "signal-v1", "protocol-v1"): 1,
        ("tenant-b", "snapshot-v1", "signal-v1", "protocol-v1"): 1,
        ("tenant-a", "snapshot-v2", "signal-v1", "protocol-v1"): 1,
    }


def test_binary_pair_aggregation_rejects_unknown_policy(spark):
    events = make_training_events(
        spark,
        [("t", "e", "u", "p", utc(2020, 1, 1), 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1")],
    )

    with pytest.raises(ValueError, match="Unsupported aggregation version"):
        aggregate_binary_pairs(events, "count_weighted_v1")


def test_user_mapping_is_zero_based_tenant_scoped_and_deterministic(spark):
    event_time = utc(2020, 1, 1)
    rows = [
        ("tenant-b", "e3", "same-user", "p3", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ("tenant-a", "e2", "user-z", "p2", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ("tenant-a", "e1", "user-a", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
    ]
    forward_pairs = aggregate_binary_pairs(make_training_events(spark, rows))
    reverse_pairs = aggregate_binary_pairs(
        make_training_events(spark, list(reversed(rows)))
    )

    def collect_mapping(pairs):
        return {
            (row["tenant_id"], row["user_id"]): row["user_index"]
            for row in build_user_mapping(pairs).collect()
        }

    expected = {
        ("tenant-a", "user-a"): 0,
        ("tenant-a", "user-z"): 1,
        ("tenant-b", "same-user"): 2,
    }
    assert collect_mapping(forward_pairs) == expected
    assert collect_mapping(reverse_pairs) == expected
    assert build_user_mapping(forward_pairs).schema["user_index"].dataType.simpleString() == "int"


def test_mappings_restart_indices_for_each_model_scope(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("t", "e1", "user-b", "product-b", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e2", "user-a", "product-a", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e3", "user-z", "product-z", event_time, 1.0, 1.0, "snapshot-v2", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)

    users = {
        (row["snapshot_id"], row["user_id"]): row["user_index"]
        for row in build_user_mapping(pairs).collect()
    }
    items = {
        (row["snapshot_id"], row["product_id"]): row["item_index"]
        for row in build_item_mapping(pairs).collect()
    }

    assert users == {
        ("snapshot-v1", "user-a"): 0,
        ("snapshot-v1", "user-b"): 1,
        ("snapshot-v2", "user-z"): 0,
    }
    assert items == {
        ("snapshot-v1", "product-a"): 0,
        ("snapshot-v1", "product-b"): 1,
        ("snapshot-v2", "product-z"): 0,
    }
    assert build_item_mapping(pairs).schema["item_index"].dataType.simpleString() == "int"


def test_attach_mapping_indices_preserves_pairs_and_uses_full_scope(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("tenant-a", "e1", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e2", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-b", "e3", "same-user", "same-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e4", "other-user", "other-product", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)
    user_mapping = build_user_mapping(pairs)
    item_mapping = build_item_mapping(pairs)

    rows = {
        (row["tenant_id"], row["user_id"], row["product_id"]): row.asDict()
        for row in attach_mapping_indices(
            pairs,
            user_mapping,
            item_mapping,
        ).collect()
    }

    assert set(rows) == {
        ("tenant-a", "other-user", "other-product"),
        ("tenant-a", "same-user", "same-product"),
        ("tenant-b", "same-user", "same-product"),
    }
    assert rows[("tenant-a", "same-user", "same-product")]["event_count"] == 2
    assert rows[("tenant-a", "same-user", "same-product")]["preference"] == 1.0
    assert rows[("tenant-a", "same-user", "same-product")]["confidence"] == 1.0
    assert rows[("tenant-a", "same-user", "same-product")]["user_index"] != rows[("tenant-b", "same-user", "same-product")]["user_index"]
    assert rows[("tenant-a", "same-user", "same-product")]["item_index"] != rows[("tenant-b", "same-user", "same-product")]["item_index"]


def test_mapping_coverage_metrics_accept_complete_mappings(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("t", "e1", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e2", "u2", "p2", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)
    users = build_user_mapping(pairs)
    items = build_item_mapping(pairs)
    indexed = attach_mapping_indices(pairs, users, items)

    metrics = calculate_mapping_coverage_metrics(
        pairs,
        users,
        items,
        indexed,
    )

    assert metrics == {
        "pair_count": 2,
        "indexed_pair_count": 2,
        "indexed_pair_parity_difference_count": 0,
        "missing_user_mapping_count": 0,
        "missing_item_mapping_count": 0,
    }
    validate_mapping_coverage_metrics(metrics)


def test_mapping_coverage_gate_rejects_silently_dropped_pair(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("t", "e1", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e2", "u2", "p2", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)
    users = build_user_mapping(pairs).filter("user_id != 'u2'")
    items = build_item_mapping(pairs)
    indexed = attach_mapping_indices(pairs, users, items)

    metrics = calculate_mapping_coverage_metrics(
        pairs,
        users,
        items,
        indexed,
    )

    assert metrics["pair_count"] == 2
    assert metrics["indexed_pair_count"] == 1
    assert metrics["indexed_pair_parity_difference_count"] == 1
    assert metrics["missing_user_mapping_count"] == 1
    assert metrics["missing_item_mapping_count"] == 0
    with pytest.raises(ValueError, match="Mapping coverage gate failed"):
        validate_mapping_coverage_metrics(metrics)


def test_train_only_artifacts_have_distinct_responsibilities(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("tenant-a", "e1", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e2", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e3", "u1", "p2", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-a", "e4", "u2", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("tenant-b", "e5", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)
    users = build_user_mapping(pairs)
    items = build_item_mapping(pairs)
    indexed = attach_mapping_indices(pairs, users, items)

    seen = {
        (row["tenant_id"], row["user_id"], row["product_id"])
        for row in build_seen_items(indexed).collect()
    }
    candidates = {
        (row["tenant_id"], row["product_id"])
        for row in build_candidate_universe(indexed).collect()
    }
    popularity = {
        (row["tenant_id"], row["product_id"]): row["positive_pair_count"]
        for row in build_item_popularity_counts(indexed).collect()
    }

    assert seen == {
        ("tenant-a", "u1", "p1"),
        ("tenant-a", "u1", "p2"),
        ("tenant-a", "u2", "p1"),
        ("tenant-b", "u1", "p1"),
    }
    assert candidates == {
        ("tenant-a", "p1"),
        ("tenant-a", "p2"),
        ("tenant-b", "p1"),
    }
    assert popularity == {
        ("tenant-a", "p1"): 2,
        ("tenant-a", "p2"): 1,
        ("tenant-b", "p1"): 1,
    }


def test_train_only_artifacts_preserve_indices_and_model_scope(spark):
    event_time = utc(2020, 1, 1)
    events = make_training_events(
        spark,
        [
            ("t", "e1", "u", "p", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    pairs = aggregate_binary_pairs(events)
    users = build_user_mapping(pairs)
    items = build_item_mapping(pairs)
    indexed = attach_mapping_indices(pairs, users, items)

    seen = build_seen_items(indexed).first().asDict()
    candidate = build_candidate_universe(indexed).first().asDict()
    popularity = build_item_popularity_counts(indexed).first().asDict()

    for row in (seen, candidate, popularity):
        for column in MAPPING_SCOPE_COLUMNS:
            assert row[column] == indexed.first()[column]
    assert seen["user_index"] == 0
    assert seen["item_index"] == 0
    assert candidate["item_index"] == 0
    assert popularity["item_index"] == 0
