from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.artifacts import (
    MODEL_INPUT_ARTIFACT_NAMES,
    calculate_model_input_signatures,
)
from shoppilot.recommendations.model_input import (
    aggregate_binary_pairs,
    attach_mapping_indices,
    build_candidate_universe,
    build_item_mapping,
    build_item_popularity_counts,
    build_seen_items,
    build_user_mapping,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderArtifactTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def make_artifacts(spark):
    event_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
    events = spark.createDataFrame(
        [
            ("t", "e1", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e2", "u1", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e3", "u2", "p1", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
            ("t", "e4", "u2", "p2", event_time, 1.0, 1.0, "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
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
    pairs = aggregate_binary_pairs(events)
    users = build_user_mapping(pairs)
    items = build_item_mapping(pairs)
    indexed = attach_mapping_indices(pairs, users, items)
    return {
        "train_pairs": pairs,
        "user_mapping": users,
        "item_mapping": items,
        "indexed_pairs": indexed,
        "seen_items": build_seen_items(indexed),
        "candidate_universe": build_candidate_universe(indexed),
        "item_popularity": build_item_popularity_counts(indexed),
    }


def test_model_input_signatures_cover_every_artifact_and_business_key(spark):
    signatures = calculate_model_input_signatures(make_artifacts(spark))

    assert set(signatures) == set(MODEL_INPUT_ARTIFACT_NAMES)
    assert {
        name: signature["row_count"] for name, signature in signatures.items()
    } == {
        "train_pairs": 3,
        "user_mapping": 2,
        "item_mapping": 2,
        "indexed_pairs": 3,
        "seen_items": 3,
        "candidate_universe": 2,
        "item_popularity": 2,
    }
    assert all(
        signature["duplicate_business_keys"] == 0
        for signature in signatures.values()
    )


def test_model_input_signatures_reject_missing_artifact(spark):
    artifacts = make_artifacts(spark)
    del artifacts["item_popularity"]

    with pytest.raises(ValueError, match="item_popularity"):
        calculate_model_input_signatures(artifacts)


def test_train_pair_signature_detects_diagnostic_value_change(spark):
    artifacts = make_artifacts(spark)
    original = calculate_model_input_signatures(artifacts)
    artifacts["train_pairs"] = artifacts["train_pairs"].withColumn(
        "event_count",
        artifacts["train_pairs"]["event_count"] + 1,
    )
    changed = calculate_model_input_signatures(artifacts)

    assert (
        original["train_pairs"]["stable_checksum"]
        != changed["train_pairs"]["stable_checksum"]
    )


def test_indexed_pair_signature_detects_user_index_change(spark):
    artifacts = make_artifacts(spark)
    original = calculate_model_input_signatures(artifacts)
    artifacts["indexed_pairs"] = artifacts["indexed_pairs"].withColumn(
        "user_index",
        artifacts["indexed_pairs"]["user_index"] + 1,
    )
    changed = calculate_model_input_signatures(artifacts)

    assert (
        original["indexed_pairs"]["stable_checksum"]
        != changed["indexed_pairs"]["stable_checksum"]
    )
