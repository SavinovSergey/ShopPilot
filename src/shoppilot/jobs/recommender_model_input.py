"""Publish reproducible recommender model-input artifacts from Gold."""

import os
from pathlib import Path
from dataclasses import asdict
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from shoppilot.recommendations.config import ModelInputConfig, calculate_config_hash
from shoppilot.recommendations.pipeline import build_model_input_bundle
from shoppilot.recommendations.artifacts import (
    MODEL_INPUT_ARTIFACT_NAMES, calculate_model_input_signatures)


METADATA_SCHEMA = StructType([
    StructField("metadata_type", StringType(), False),
    StructField("entity", StringType(), False),
    StructField("metric_name", StringType(), False),
    StructField("metric_value", StringType(), True),
])

OUTPUT_PARTITIONS = {
    "train_pairs": 4,
    "user_mapping": 4,
    "item_mapping": 2,
    "indexed_pairs": 4,
    "seen_items": 4,
    "candidate_universe": 1,
    "item_popularity": 1,
}

DEFAULT_SNAPSHOT_ID = "amazon-all-beauty-2023-v1"
DEFAULT_SIGNAL_VERSION = "rating_implicit_uniform_v1"
DEFAULT_PROTOCOL_VERSION = "global_calendar_v1"
DEFAULT_VALIDATION_CUTOFF = "2021-01-01"


def _as_metadata_value(value):
    return None if value is None else str(value)

def build_metadata_rows(
    config: ModelInputConfig,
    model_input_bundle: dict,
) -> list[tuple]:
    rows = [
        (
            "config",
            "model_input",
            "config_hash",
            model_input_bundle["config_hash"],
        )
    ]
    for field_name, value in sorted(asdict(config).items()):
        rows.append(
            (
                "config",
                "model_input",
                field_name,
                _as_metadata_value(value),
            )
        )
    for entity, metrics in sorted(model_input_bundle["quality_metrics"].items()):
        for metric_name, value in sorted(metrics.items()):
            rows.append(
                (
                    "quality",
                    entity,
                    metric_name,
                    _as_metadata_value(value),
                )
            )
    for artifact_name, signature in sorted(model_input_bundle["signatures"].items()):
        for metric_name, value in sorted(signature.items()):
            rows.append(
                (
                    "signature",
                    artifact_name,
                    metric_name,
                    _as_metadata_value(value),
                )
            )
    return rows


def is_complete(path: Path) -> bool:
    """Return True if the path exists and is not empty."""
    return path.is_dir() and (path / "_SUCCESS").exists()


def run_recommender_model_input_job(
    spark,
    config: ModelInputConfig,
    project_root=".",
) -> dict:
    """Build, validate and publish one versioned model-input snapshot."""

    # region Paths
    # Gold inputs
    signals_path = os.path.join(
        project_root,
        "data/lake/gold/snapshots",
        config.snapshot_id,
        "interaction_signals",
        config.signal_version
    )
    assignments_path = os.path.join(
        project_root,
        "data/lake/gold/snapshots",
        config.snapshot_id,
        "evaluation_assignments",
        config.protocol_version
    )

    config_hash = calculate_config_hash(config)

    project_root = Path(project_root)

    # Outputs
    output_root = (
        project_root
        / "data/lake/model_inputs"
        / config.snapshot_id
        / config_hash
    )
    artifact_paths = {
        name: output_root / "artifacts" / name
        for name in MODEL_INPUT_ARTIFACT_NAMES
    }
    metadata_path = output_root / "metadata"
    # endregion Paths

    # region Read inputs
    signals_df = spark.read.parquet(signals_path)
    assignments_df = spark.read.parquet(assignments_path)
    # endregion Read inputs

    model_input_bundle = build_model_input_bundle(signals_df, assignments_df, config)
    metadata_rows = build_metadata_rows(config, model_input_bundle)
    metadata_df = spark.createDataFrame(metadata_rows, METADATA_SCHEMA)


    # region Check for existing and complete artifacts
    published_paths = {
        **artifact_paths,
        "metadata": metadata_path,
    }
    existing = {
        name: path.exists()
        for name, path in published_paths.items()
    }
    complete = {
        name: is_complete(path)
        for name, path in published_paths.items()
    }

    previous_signatures = None
    if any(existing.values()):
        if not all(complete.values()):
            raise RuntimeError("Previous model-input publication is incomplete")
        previous_artifacts = {
            name: spark.read.parquet(str(path))
            for name, path in artifact_paths.items()
        }
        previous_signatures = calculate_model_input_signatures(
            previous_artifacts
        )
        if previous_signatures != model_input_bundle["signatures"]:
            raise RuntimeError(
                "Previous model-input publication is different"
            )
    # endregion Check for existing and complete artifacts

    for name, path in artifact_paths.items():
        (
            model_input_bundle["artifacts"][name]
            .coalesce(OUTPUT_PARTITIONS[name])
            .write
            .mode("overwrite")
            .partitionBy("tenant_id")
            .parquet(str(path))
        )
    metadata_df.coalesce(1).write.mode("overwrite").parquet(
        str(metadata_path)
    )

    saved_artifacts = {
        name: spark.read.parquet(str(path))
        for name, path in artifact_paths.items()
    }
    saved_signatures = calculate_model_input_signatures(
        saved_artifacts
    )

    if saved_signatures != model_input_bundle["signatures"]:
        raise RuntimeError(
            "Saved model-input artifacts differ from in-memory signatures"
        )

    return {
        "config_hash": model_input_bundle["config_hash"],
        "quality_metrics": model_input_bundle["quality_metrics"],
        "current_signatures": model_input_bundle["signatures"],
        "previous_signatures": previous_signatures,
    }


def main():
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("Recommender Model Input Job")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    config = ModelInputConfig(
        snapshot_id=DEFAULT_SNAPSHOT_ID,
        signal_version=DEFAULT_SIGNAL_VERSION,
        protocol_version=DEFAULT_PROTOCOL_VERSION,
        cutoff=DEFAULT_VALIDATION_CUTOFF,
    )
    try:
        result = run_recommender_model_input_job(spark, config)
        print("Model-input config hash:", result["config_hash"])
        print("Model-input quality metrics:", result["quality_metrics"])
        print("Model-input signatures:", result["current_signatures"])
        if result["previous_signatures"] is None:
            print("Previous model-input publication not found")
        else:
            print("Model-input rerun signatures match previous publication")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
