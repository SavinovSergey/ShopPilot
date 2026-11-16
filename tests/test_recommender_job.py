from datetime import datetime, timezone

import pytest
from pyspark.sql import SparkSession

from shoppilot.jobs.recommender_model_input import (
    run_recommender_model_input_job,
)
from shoppilot.recommendations.artifacts import MODEL_INPUT_ARTIFACT_NAMES
from shoppilot.recommendations.config import (
    ModelInputConfig,
    calculate_config_hash,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderModelInputJobTests")
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


def write_gold_inputs(spark, project_root, config):
    snapshot_root = (
        project_root
        / "data/lake/gold/snapshots"
        / config.snapshot_id
    )
    signals = spark.createDataFrame(
        [
            ("t", "e1", "u1", "p1", utc(2020, 1, 1), config.signal_version, "positive", 1.0, 1.0, True, config.snapshot_id),
            ("t", "e2", "u1", "p1", utc(2020, 2, 1), config.signal_version, "positive", 1.0, 1.0, True, config.snapshot_id),
            ("t", "e3", "u2", "p2", utc(2020, 3, 1), config.signal_version, "positive", 1.0, 1.0, True, config.snapshot_id),
            ("t", "validation", "u1", "p2", utc(2021, 2, 1), config.signal_version, "positive", 1.0, 1.0, True, config.snapshot_id),
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
            ("t", "e1", "u1", "p1", utc(2020, 1, 1), config.signal_version, config.protocol_version, "train", True, config.snapshot_id),
            ("t", "e2", "u1", "p1", utc(2020, 2, 1), config.signal_version, config.protocol_version, "train", True, config.snapshot_id),
            ("t", "e3", "u2", "p2", utc(2020, 3, 1), config.signal_version, config.protocol_version, "train", True, config.snapshot_id),
            ("t", "validation", "u1", "p2", utc(2021, 2, 1), config.signal_version, config.protocol_version, "validation", True, config.snapshot_id),
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
    signals.write.parquet(
        str(snapshot_root / "interaction_signals" / config.signal_version)
    )
    assignments.write.parquet(
        str(
            snapshot_root
            / "evaluation_assignments"
            / config.protocol_version
        )
    )


def test_recommender_model_input_job_publishes_metadata_and_checks_rerun(
    spark,
    tmp_path,
):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    write_gold_inputs(spark, tmp_path, config)
    config_hash = calculate_config_hash(config)
    output_root = (
        tmp_path
        / "data/lake/model_inputs"
        / config.snapshot_id
        / config_hash
    )

    first = run_recommender_model_input_job(
        spark,
        config,
        project_root=tmp_path,
    )

    assert first["config_hash"] == config_hash
    assert first["previous_signatures"] is None
    assert first["current_signatures"]["train_pairs"]["row_count"] == 2
    for artifact_name in MODEL_INPUT_ARTIFACT_NAMES:
        assert (output_root / "artifacts" / artifact_name).is_dir()

    metadata = spark.read.parquet(str(output_root / "metadata"))
    metadata_rows = {
        (
            row["metadata_type"],
            row["entity"],
            row["metric_name"],
        ): row["metric_value"]
        for row in metadata.collect()
    }
    assert metadata_rows[("config", "model_input", "config_hash")] == config_hash
    assert metadata_rows[("signature", "train_pairs", "row_count")] == "2"
    assert metadata_rows[("quality", "leakage", "future_role_overlap_count")] == "0"

    second = run_recommender_model_input_job(
        spark,
        config,
        project_root=tmp_path,
    )

    assert second["previous_signatures"] == second["current_signatures"]
