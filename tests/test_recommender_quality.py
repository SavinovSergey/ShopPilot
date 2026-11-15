from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.quality import (
    calculate_training_leakage_metrics,
    validate_training_leakage_metrics,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderQualityTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def config():
    return ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )


def make_training_events(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        event_id string,
        user_id string,
        product_id string,
        event_time timestamp,
        snapshot_id string,
        signal_version string,
        protocol_version string
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
        snapshot_id string,
        signal_version string,
        protocol_version string,
        role string
        """,
    )


def utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_training_leakage_metrics_accept_clean_train_selection(
    spark,
    config,
):
    training_events = make_training_events(
        spark,
        [
            ("t", "train", "u1", "p1", utc(2020, 1, 1), "snapshot-v1", "signal-v1", "protocol-v1"),
        ],
    )
    assignments = make_assignments(
        spark,
        [
            ("t", "train", "u1", "p1", utc(2020, 1, 1), "snapshot-v1", "signal-v1", "protocol-v1", "train"),
            ("t", "validation", "u1", "p2", utc(2021, 1, 1), "snapshot-v1", "signal-v1", "protocol-v1", "validation"),
            ("t", "test", "u1", "p3", utc(2022, 1, 1), "snapshot-v1", "signal-v1", "protocol-v1", "test"),
        ],
    )

    metrics = calculate_training_leakage_metrics(
        training_events,
        assignments,
        config,
    )

    assert metrics == {
        "training_event_count": 1,
        "future_role_overlap_count": 0,
        "at_or_after_cutoff_count": 0,
        "wrong_scope_count": 0,
        "missing_training_assignment_count": 0,
    }
    validate_training_leakage_metrics(metrics)


def test_training_leakage_gate_rejects_validation_id_in_train(
    spark,
    config,
):
    leaked_event = (
        "t", "leaked", "u", "p", utc(2020, 6, 1),
        "snapshot-v1", "signal-v1", "protocol-v1",
    )
    training_events = make_training_events(spark, [leaked_event])
    assignments = make_assignments(
        spark,
        [
            (*leaked_event, "validation"),
            (*leaked_event, "test"),
        ],
    )

    metrics = calculate_training_leakage_metrics(
        training_events,
        assignments,
        config,
    )

    assert metrics["future_role_overlap_count"] == 1
    with pytest.raises(ValueError, match="future_role_overlap_count: 1"):
        validate_training_leakage_metrics(metrics)


def test_training_leakage_gate_treats_cutoff_as_exclusive(
    spark,
    config,
):
    boundary_event = (
        "t", "boundary", "u", "p", utc(2021, 1, 1),
        "snapshot-v1", "signal-v1", "protocol-v1",
    )
    training_events = make_training_events(spark, [boundary_event])
    assignments = make_assignments(
        spark,
        [(*boundary_event, "train")],
    )

    metrics = calculate_training_leakage_metrics(
        training_events,
        assignments,
        config,
    )

    assert metrics["at_or_after_cutoff_count"] == 1
    with pytest.raises(ValueError, match="at_or_after_cutoff_count: 1"):
        validate_training_leakage_metrics(metrics)
