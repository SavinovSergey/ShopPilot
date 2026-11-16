from dataclasses import replace
from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.artifacts import MODEL_INPUT_ARTIFACT_NAMES
from shoppilot.recommendations.config import (
    ModelInputConfig,
    calculate_config_hash,
)
from shoppilot.recommendations.pipeline import build_model_input_bundle


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderPipelineTests")
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


def test_build_model_input_bundle_connects_artifacts_gates_and_metadata(spark):
    signals = spark.createDataFrame(
        [
            ("t", "e1", "u1", "p1", utc(2020, 1, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
            ("t", "e2", "u1", "p1", utc(2020, 2, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
            ("t", "e3", "u2", "p2", utc(2020, 3, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
            ("t", "validation", "u1", "p2", utc(2021, 2, 1), "signal-v1", "positive", 1.0, 1.0, True, "snapshot-v1"),
        ],
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
    assignments = spark.createDataFrame(
        [
            ("t", "e1", "u1", "p1", utc(2020, 1, 1), "signal-v1", "protocol-v1", "train", True, "snapshot-v1"),
            ("t", "e2", "u1", "p1", utc(2020, 2, 1), "signal-v1", "protocol-v1", "train", True, "snapshot-v1"),
            ("t", "e3", "u2", "p2", utc(2020, 3, 1), "signal-v1", "protocol-v1", "train", True, "snapshot-v1"),
            ("t", "validation", "u1", "p2", utc(2021, 2, 1), "signal-v1", "protocol-v1", "validation", True, "snapshot-v1"),
        ],
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
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )

    bundle = build_model_input_bundle(signals, assignments, config)

    assert set(bundle) == {
        "artifacts",
        "quality_metrics",
        "signatures",
        "config_hash",
    }
    assert set(bundle["artifacts"]) == set(MODEL_INPUT_ARTIFACT_NAMES)
    assert set(bundle["signatures"]) == set(MODEL_INPUT_ARTIFACT_NAMES)
    assert bundle["config_hash"] == calculate_config_hash(config)

    assert bundle["quality_metrics"]["leakage"] == {
        "training_event_count": 3,
        "future_role_overlap_count": 0,
        "at_or_after_cutoff_count": 0,
        "wrong_scope_count": 0,
        "missing_training_assignment_count": 0,
    }
    assert bundle["quality_metrics"]["mapping_coverage"] == {
        "pair_count": 2,
        "indexed_pair_count": 2,
        "indexed_pair_parity_difference_count": 0,
        "missing_user_mapping_count": 0,
        "missing_item_mapping_count": 0,
    }

    pair_rows = {
        (row["user_id"], row["product_id"]): row["event_count"]
        for row in bundle["artifacts"]["train_pairs"].collect()
    }
    assert pair_rows == {("u1", "p1"): 2, ("u2", "p2"): 1}
    assert bundle["signatures"]["train_pairs"]["row_count"] == 2


@pytest.mark.parametrize(
    ("field_name", "unsupported_value"),
    [
        ("aggregation_version", "count_weighted_v1"),
        ("candidate_policy_version", "all_catalog_items_v1"),
        ("popularity_policy_version", "event_count_v1"),
        ("training_role", "test"),
    ],
)
def test_build_model_input_bundle_rejects_unsupported_semantic_config(
    field_name,
    unsupported_value,
):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    config = replace(config, **{field_name: unsupported_value})

    with pytest.raises(ValueError, match=field_name):
        build_model_input_bundle(None, None, config)
