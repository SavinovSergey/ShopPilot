from datetime import date
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType, StructField, StructType

from shoppilot.data.gold.document_quality import (
    calculate_product_document_metrics,
    validate_product_document_metrics,
)
from shoppilot.data.gold.documents import (
    PRODUCT_DOCUMENT_OUTPUT_COLUMNS,
    build_product_documents,
)
from shoppilot.data.gold.quality import (
    calculate_assignment_metrics,
    calculate_fact_metrics,
    calculate_signal_metrics,
    calculate_user_feature_metrics,
    validate_gold_v2_metrics,
)
from shoppilot.data.gold.readiness import validate_silver_readiness
from shoppilot.data.gold.transforms import (
    DEFAULT_PROTOCOL_VERSION,
    DEFAULT_SIGNAL_VERSION,
    assign_global_calendar_roles,
    attach_current_catalog_status,
    build_rating_signals,
    build_user_features_at_cutoff,
    build_user_item_events,
)
from shoppilot.data.signatures import (
    calculate_dataset_signature,
    metrics_to_rows,
)


DEFAULT_SNAPSHOT_ID = "amazon-all-beauty-2023-v1"
FEATURE_CUTOFFS = ("2021-01-01", "2022-01-01", "2023-01-01")

# This snapshot is a few hundred MB and currently contains one tenant.  The
# upstream temporal unions carry hundreds of shuffle partitions, which would
# otherwise become hundreds of tiny Parquet files.  coalesce only changes the
# write boundary; it does not change logical rows or dataset signatures.
OUTPUT_PARTITIONS = {
    "user_item_events": 5,
    "interaction_signals": 5,
    "evaluation_assignments": 8,
    "user_features": 4,
    "product_documents": 4,
}

EVENT_COLUMNS = (
    "tenant_id",
    "event_id",
    "user_id",
    "product_id",
    "event_type",
    "event_time",
    "rating",
    "verified_purchase",
    "helpful_vote",
    "source",
    "source_record_id",
    "is_missing_current_catalog",
    "current_catalog_is_active",
    "snapshot_id",
)

SIGNAL_COLUMNS = (
    "tenant_id",
    "event_id",
    "user_id",
    "product_id",
    "event_time",
    "signal_name",
    "signal_version",
    "feedback_class",
    "preference",
    "confidence",
    "is_training_eligible",
    "snapshot_id",
)

ASSIGNMENT_COLUMNS = (
    "tenant_id",
    "event_id",
    "user_id",
    "product_id",
    "event_time",
    "signal_version",
    "evaluation_protocol",
    "protocol_version",
    "role",
    "cutoff_time",
    "target_bucket_id",
    "is_training_eligible",
    "is_evaluation_target",
    "history_depth_before_cutoff",
    "user_availability_state",
    "item_availability_state",
    "snapshot_id",
)

USER_FEATURE_COLUMNS = (
    "tenant_id",
    "user_id",
    "all_interaction_count",
    "all_distinct_event_time_count",
    "positive_interaction_count",
    "positive_distinct_event_time_count",
    "neutral_interaction_count",
    "negative_interaction_count",
    "first_event_time",
    "last_event_time",
    "first_positive_event_time",
    "last_positive_event_time",
    "feature_version",
    "feature_cutoff",
    "snapshot_id",
)


def _output_paths(project_root: Path, snapshot_id: str) -> dict:
    snapshot_root = project_root / "data/lake/gold/snapshots" / snapshot_id
    return {
        "user_item_events": snapshot_root / "user_item_events",
        "interaction_signals": (
            snapshot_root
            / "interaction_signals"
            / DEFAULT_SIGNAL_VERSION
        ),
        "evaluation_assignments": (
            snapshot_root
            / "evaluation_assignments"
            / DEFAULT_PROTOCOL_VERSION
        ),
        "user_features": snapshot_root / "user_features",
        "product_documents": snapshot_root / "product_documents",
    }


def _signature_specs() -> dict:
    return {
        "user_item_events": (
            ["tenant_id", "event_id"],
            list(EVENT_COLUMNS),
        ),
        "interaction_signals": (
            ["tenant_id", "event_id", "signal_version"],
            list(SIGNAL_COLUMNS),
        ),
        "evaluation_assignments": (
            ["tenant_id", "event_id", "protocol_version"],
            list(ASSIGNMENT_COLUMNS),
        ),
        "user_features": (
            ["tenant_id", "user_id", "feature_cutoff"],
            list(USER_FEATURE_COLUMNS),
        ),
        "product_documents": (
            ["tenant_id", "product_id"],
            [*PRODUCT_DOCUMENT_OUTPUT_COLUMNS, "snapshot_id"],
        ),
    }


def _dataset_signatures(datasets: dict) -> dict:
    signatures = {}
    for name, (business_keys, checksum_columns) in _signature_specs().items():
        signatures[name] = calculate_dataset_signature(
            datasets[name],
            business_keys,
            checksum_columns,
        )
    return signatures


def _read_outputs(spark, paths: dict) -> dict:
    return {name: spark.read.parquet(str(path)) for name, path in paths.items()}


def _prepare_for_write(name: str, dataframe):
    return dataframe.coalesce(OUTPUT_PARTITIONS[name])


def _target_slice_metrics(assignments_df) -> dict:
    rows = (
        assignments_df.filter(F.col("is_evaluation_target"))
        .groupBy(
            "role",
            "user_availability_state",
            "item_availability_state",
        )
        .agg(
            F.count("*").alias("event_count"),
            F.countDistinct(F.struct("tenant_id", "user_id")).alias(
                "user_count"
            ),
        )
        .collect()
    )
    metrics = {}
    for row in rows:
        prefix = "__".join(
            (
                row["role"],
                row["user_availability_state"],
                row["item_availability_state"],
            )
        )
        metrics[f"{prefix}__event_count"] = row["event_count"]
        metrics[f"{prefix}__user_count"] = row["user_count"]
    return metrics


def run_gold_v2_job(
    spark,
    run_date: str,
    project_root=".",
    snapshot_id: str = DEFAULT_SNAPSHOT_ID,
) -> dict:
    if not snapshot_id or "/" in snapshot_id or ".." in snapshot_id:
        raise ValueError("snapshot_id must be a non-empty path-safe name")

    project_root = Path(project_root)
    products_path = project_root / "data/lake/silver/products"
    reviews_path = project_root / "data/lake/silver/reviews"
    output_paths = _output_paths(project_root, snapshot_id)
    metrics_path = (
        project_root
        / "data/lake/quality/gold_v2"
        / snapshot_id
        / "metrics"
        / run_date
    )

    path_states = {name: path.exists() for name, path in output_paths.items()}
    if any(path_states.values()) and not all(path_states.values()):
        existing = sorted(name for name, exists in path_states.items() if exists)
        missing = sorted(name for name, exists in path_states.items() if not exists)
        raise RuntimeError(
            "Previous Gold v2 run is incomplete. "
            f"Existing outputs: {existing}; missing outputs: {missing}"
        )

    products = spark.read.parquet(str(products_path))
    reviews = spark.read.parquet(str(reviews_path))
    readiness_metrics = validate_silver_readiness(products, reviews)

    events = (
        attach_current_catalog_status(
            build_user_item_events(reviews, snapshot_id),
            products,
        )
        .select(*EVENT_COLUMNS)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    signals = (
        build_rating_signals(events)
        .select(*SIGNAL_COLUMNS)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    assignments = (
        assign_global_calendar_roles(signals)
        .select(*ASSIGNMENT_COLUMNS)
        .persist(StorageLevel.MEMORY_AND_DISK)
    )

    feature_frames = [
        build_user_features_at_cutoff(signals, cutoff)
        for cutoff in FEATURE_CUTOFFS
    ]
    user_features = (
        feature_frames[0]
        .unionByName(feature_frames[1])
        .unionByName(feature_frames[2])
        .persist(StorageLevel.MEMORY_AND_DISK)
    )
    product_documents = (
        build_product_documents(products)
        .select(*PRODUCT_DOCUMENT_OUTPUT_COLUMNS)
        .withColumn("snapshot_id", F.lit(snapshot_id))
        .persist(StorageLevel.DISK_ONLY)
    )

    datasets = {
        "user_item_events": events,
        "interaction_signals": signals,
        "evaluation_assignments": assignments,
        "user_features": user_features,
        "product_documents": product_documents,
    }

    try:
        for dataframe in datasets.values():
            dataframe.count()

        quality_metrics = {}
        quality_metrics.update(calculate_fact_metrics(reviews, events))
        quality_metrics.update(calculate_signal_metrics(events, signals))
        quality_metrics.update(
            calculate_assignment_metrics(signals, assignments)
        )
        quality_metrics.update(calculate_user_feature_metrics(user_features))
        validate_gold_v2_metrics(quality_metrics)
        if quality_metrics["feature_cutoff_count"] != len(FEATURE_CUTOFFS):
            raise ValueError(
                "Expected user features for "
                f"{len(FEATURE_CUTOFFS)} cutoffs, got "
                f"{quality_metrics['feature_cutoff_count']}"
            )

        product_metrics = calculate_product_document_metrics(product_documents)
        validate_product_document_metrics(product_metrics)
        product_metrics["missing_price_count"] = product_documents.filter(
            F.col("price").isNull()
        ).count()

        slice_metrics = _target_slice_metrics(assignments)
        current_signatures = _dataset_signatures(datasets)

        previous_signatures = None
        if all(path_states.values()):
            previous_signatures = _dataset_signatures(
                _read_outputs(spark, output_paths)
            )
            if current_signatures != previous_signatures:
                raise RuntimeError(
                    "Gold v2 rerun signature mismatch. "
                    f"Current: {current_signatures}; "
                    f"previous: {previous_signatures}"
                )

        for name, dataframe in datasets.items():
            (
                _prepare_for_write(name, dataframe)
                .write.mode("overwrite")
                .partitionBy("tenant_id")
                .parquet(str(output_paths[name]))
            )

        metric_rows = []
        metric_rows.extend(
            metrics_to_rows("silver_readiness", run_date, readiness_metrics)
        )
        metric_rows.extend(
            metrics_to_rows("gold_v2_quality", run_date, quality_metrics)
        )
        metric_rows.extend(
            metrics_to_rows("product_documents", run_date, product_metrics)
        )
        metric_rows.extend(
            metrics_to_rows("evaluation_slices", run_date, slice_metrics)
        )
        for name, signature in current_signatures.items():
            metric_rows.extend(
                metrics_to_rows(f"signature_{name}", run_date, signature)
            )

        metric_schema = StructType(
            [
                StructField("entity", StringType(), False),
                StructField("run_date", StringType(), False),
                StructField("metric_name", StringType(), False),
                StructField("metric_value", StringType(), True),
            ]
        )
        metrics_df = spark.createDataFrame(metric_rows, metric_schema)
        metrics_df.write.mode("overwrite").parquet(str(metrics_path))

        saved_signatures = _dataset_signatures(
            _read_outputs(spark, output_paths)
        )
        if saved_signatures != current_signatures:
            raise RuntimeError(
                "Saved Gold v2 outputs differ from in-memory signatures"
            )

        return {
            "snapshot_id": snapshot_id,
            "readiness_metrics": readiness_metrics,
            "quality_metrics": quality_metrics,
            "product_metrics": product_metrics,
            "slice_metrics": slice_metrics,
            "current_signatures": current_signatures,
            "previous_signatures": previous_signatures,
        }
    finally:
        for dataframe in datasets.values():
            dataframe.unpersist()


def main():
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("Gold v2 Job")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    try:
        result = run_gold_v2_job(spark, date.today().isoformat())
        print("Gold v2 snapshot:", result["snapshot_id"])
        print("Gold v2 readiness metrics:", result["readiness_metrics"])
        print("Gold v2 quality metrics:", result["quality_metrics"])
        print("Gold v2 product metrics:", result["product_metrics"])
        print("Gold v2 evaluation slices:", result["slice_metrics"])
        print("Gold v2 signatures:", result["current_signatures"])
        if result["previous_signatures"] is None:
            print("Previous Gold v2 outputs not found; rerun comparison skipped")
        else:
            print("Gold v2 rerun signatures match previous outputs")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
