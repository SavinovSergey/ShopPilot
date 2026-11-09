"""Inspect physical plans, skew and Parquet layout for the Gold pipeline."""

from collections import Counter
import argparse
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from shoppilot.jobs.gold import DEFAULT_SNAPSHOT_ID, FEATURE_CUTOFFS
from shoppilot.data.gold.transforms import (
    assign_global_calendar_roles,
    attach_current_catalog_status,
    build_rating_signals,
    build_user_features_at_cutoff,
    build_user_item_events,
)


PLAN_NODES = (
    "Exchange",
    "SortMergeJoin",
    "BroadcastHashJoin",
    "HashAggregate",
    "SortAggregate",
    "Union",
    "InMemoryTableScan",
    "FileScan",
)


def plan_node_counts(dataframe) -> dict:
    plan = dataframe._jdf.queryExecution().executedPlan().toString()
    counts = Counter()
    for node in PLAN_NODES:
        counts[node] = plan.count(node)
    return dict(counts)


def parquet_layout(path: Path) -> dict:
    files = list(path.rglob("*.parquet"))
    sizes = [file.stat().st_size for file in files]
    return {
        "file_count": len(files),
        "total_bytes": sum(sizes),
        "min_bytes": min(sizes) if sizes else 0,
        "max_bytes": max(sizes) if sizes else 0,
    }


def build_audit_dataframes(spark, project_root=".") -> dict:
    root = Path(project_root)
    products = spark.read.parquet(str(root / "data/lake/silver/products"))
    reviews = spark.read.parquet(str(root / "data/lake/silver/reviews"))
    events = attach_current_catalog_status(
        build_user_item_events(reviews, DEFAULT_SNAPSHOT_ID),
        products,
    ).persist(StorageLevel.MEMORY_AND_DISK)
    events.count()
    signals = build_rating_signals(events).persist(StorageLevel.MEMORY_AND_DISK)
    signals.count()
    assignments = assign_global_calendar_roles(signals)
    feature_frames = [
        build_user_features_at_cutoff(signals, cutoff)
        for cutoff in FEATURE_CUTOFFS
    ]
    user_features = feature_frames[0].unionByName(
        feature_frames[1]
    ).unionByName(feature_frames[2])
    return {
        "events": events,
        "signals": signals,
        "assignments": assignments,
        "user_features": user_features,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formatted",
        action="store_true",
        help="Print the full formatted plans after the compact audit.",
    )
    args = parser.parse_args()
    spark = (
        SparkSession.builder.master("local[4]")
        .appName("Gold v2 Performance Audit")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    dataframes = None
    try:
        dataframes = build_audit_dataframes(spark)
        print("Spark AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))
        print(
            "Auto broadcast threshold:",
            spark.conf.get("spark.sql.autoBroadcastJoinThreshold"),
        )
        print(
            "Shuffle partitions:",
            spark.conf.get("spark.sql.shuffle.partitions"),
        )
        for name, dataframe in dataframes.items():
            print(f"{name} input partitions:", dataframe.rdd.getNumPartitions())
            print(f"{name} plan nodes:", plan_node_counts(dataframe))

        snapshot_root = (
            Path("data/lake/gold/snapshots") / DEFAULT_SNAPSHOT_ID
        )
        output_paths = {
            "user_item_events": snapshot_root / "user_item_events",
            "interaction_signals": (
                snapshot_root
                / "interaction_signals"
                / "rating_implicit_uniform_v1"
            ),
            "evaluation_assignments": (
                snapshot_root
                / "evaluation_assignments"
                / "global_calendar_v1"
            ),
            "user_features": snapshot_root / "user_features",
            "product_documents": snapshot_root / "product_documents",
        }
        for name, path in output_paths.items():
            print(f"{name} parquet layout:", parquet_layout(path))

        user_counts = dataframes["signals"].groupBy(
            "tenant_id", "user_id"
        ).count()
        item_counts = (
            dataframes["signals"]
            .filter(F.col("is_training_eligible"))
            .groupBy("tenant_id", "product_id")
            .count()
        )
        print(
            "events per user p50/p90/p99:",
            user_counts.approxQuantile("count", [0.5, 0.9, 0.99], 0.001),
        )
        print(
            "events per user max:",
            user_counts.agg(F.max("count")).first()[0],
        )
        print(
            "positive events per item p50/p90/p99:",
            item_counts.approxQuantile("count", [0.5, 0.9, 0.99], 0.001),
        )
        print(
            "positive events per item max:",
            item_counts.agg(F.max("count")).first()[0],
        )

        pruned_assignments = (
            spark.read.parquet(str(output_paths["evaluation_assignments"]))
            .filter(F.col("tenant_id") == "beauty-demo")
            .select("event_id")
        )
        pruning_plan = (
            pruned_assignments._jdf.queryExecution().executedPlan().toString()
        )
        print(
            "tenant partition pruning detected:",
            "PartitionFilters" in pruning_plan and "beauty-demo" in pruning_plan,
        )

        if args.formatted:
            print("\nFormatted plan: current-catalog event join")
            dataframes["events"].explain(mode="formatted")
            print("\nFormatted plan: global calendar assignments")
            dataframes["assignments"].explain(mode="formatted")
            print("\nFormatted plan: point-in-time user features")
            dataframes["user_features"].explain(mode="formatted")
    finally:
        if dataframes is not None:
            dataframes["signals"].unpersist()
            dataframes["events"].unpersist()
        spark.stop()


if __name__ == "__main__":
    main()
