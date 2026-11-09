from datetime import date
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from shoppilot.data.silver.transforms import (
    build_silver_products,
    build_silver_reviews,
    select_review_evidence,
)


PRODUCT_REQUIRED_COLUMNS = [
    "tenant_id",
    "product_id",
    "title",
    "category",
    "brand",
    "price",
    "features",
    "description",
    "details",
    "updated_at",
    "is_active",
]

REVIEW_REQUIRED_COLUMNS = [
    "tenant_id",
    "review_id",
    "product_id",
    "user_id",
    "rating",
    "title",
    "text",
    "verified_purchase",
    "helpful_vote",
    "event_time",
]


def validate_quality_metrics(metrics: dict) -> None:
    invalid_metrics = []
    for key, value in metrics.items():
        if value > 0:
            invalid_metrics.append(f"{key}: {value}")
    if invalid_metrics:
        error_details = "\n".join(invalid_metrics)
        raise ValueError(f"Invalid metrics: {error_details}")


def validate_required_columns(
    dataframe,
    entity: str,
    required_columns: list[str],
) -> None:
    missing_columns = sorted(
        set(required_columns) - set(dataframe.columns)
    )
    if missing_columns:
        missing_columns_list = ", ".join(missing_columns)
        raise ValueError(
            f"{entity} is missing required columns: "
            f"{missing_columns_list}"
        )


def metrics_to_rows(entity: str, run_date: str, metrics: dict) -> list[tuple]:
    return [
        (entity, run_date, metric_name, metric_value)
        for metric_name, metric_value in metrics.items()
    ]


def calculate_dataset_signature(
    dataframe,
    business_key_columns: list[str],
    checksum_columns: list[str],
) -> dict:
    row_count = dataframe.count()
    distinct_business_keys = (
        dataframe
        .select(*business_key_columns)
        .distinct()
        .count()
    )
    duplicate_business_keys = row_count - distinct_business_keys

    serialized_value = F.to_json(
        F.struct(*[F.col(name) for name in checksum_columns]),
        {"ignoreNullFields": "false"},
    )
    checksum_row = (
        dataframe
        .agg(
            F.coalesce(
                F.sum(
                    F.xxhash64(serialized_value).cast("decimal(38, 0)")
                ),
                F.lit(0).cast("decimal(38, 0)"),
            ).alias("stable_checksum")
        )
        .first()
    )
    checksum = checksum_row["stable_checksum"]

    return {
        "row_count": row_count,
        "distinct_business_keys": distinct_business_keys,
        "duplicate_business_keys": duplicate_business_keys,
        "stable_checksum": str(checksum),
    }


def main():
    run_date = date.today().isoformat()

    reviews_bronze_path = "./data/lake/bronze/reviews"
    metadata_bronze_path = "./data/lake/bronze/metadata"
    products_path = "./data/lake/silver/products"
    reviews_path = "./data/lake/silver/reviews"
    evidence_path = "./data/lake/gold/review_evidence"
    quality_metrics_path = (
        f"./data/lake/quality/silver/metrics/{run_date}"
    )

    product_quarantine_path = (
        f"./data/lake/quality/silver/products/"
        f"{run_date}/quarantine"
    )

    review_orphans_path = (
        f"./data/lake/quality/silver/reviews/"
        f"{run_date}/orphans"
    )

    spark = (
        SparkSession.builder
        .appName("Silver Job")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.log.level", "WARN")
        .getOrCreate()
    )

    reviews_bronze = spark.read.parquet(reviews_bronze_path)
    metadata_bronze = spark.read.parquet(metadata_bronze_path)

    # region: reviews
    silver_reviews = build_silver_reviews(reviews_bronze)
    row_count_before_dedup = reviews_bronze.count()
    row_count_after_dedup = silver_reviews.count()

    duplicate_review_keys = (
        silver_reviews
        .groupBy(
            "tenant_id",
            "user_id",
            "product_id",
            "event_time",
        )
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    review_dedup_metrics = {
        "row_count_before_dedup": row_count_before_dedup,
        "row_count_after_dedup": row_count_after_dedup,
        "rows_removed_by_dedup": (
            row_count_before_dedup - row_count_after_dedup
        ),
        "duplicate_review_keys": duplicate_review_keys,
    }
    review_violation_metrics = (
        silver_reviews.agg(
            F.sum(
                F.col("tenant_id").isNull().cast("long")
            ).alias("missing_tenant_id"),
            F.sum(
                F.col("review_id").isNull().cast("long")
            ).alias("missing_review_id"),
            F.sum(
                F.col("product_id").isNull().cast("long")
            ).alias("missing_product_id"),
            F.sum(
                F.col("user_id").isNull().cast("long")
            ).alias("missing_user_id"),
            F.sum(
                F.col("event_time").isNull().cast("long")
            ).alias("missing_event_time"),
            F.sum(
                F.col("source_record_id").isNull().cast("long")
            ).alias("missing_source_record_id"),
            F.sum(
                (
                    F.col("rating").isNull()
                    | (F.col("rating") < 1)
                    | (F.col("rating") > 5)
                ).cast("long")
            ).alias("invalid_rating"),
            F.sum(
                (F.col("helpful_vote") < 0).cast("long")
            ).alias("negative_helpful_vote"),
        )
        .first()
        .asDict()
    )

    assert row_count_before_dedup >= row_count_after_dedup
    assert duplicate_review_keys == 0
    validate_quality_metrics(review_violation_metrics)
    # endregion: reviews

    # region: products
    silver_products = build_silver_products(metadata_bronze)

    product_source_metrics = (
        metadata_bronze.agg(
            F.count("*").alias("row_count_before_dedup"),
            F.sum(
                (
                    F.col("price").isNotNull()
                    & (F.col("price") <= 0)
                ).cast("long")
            ).alias("invalid_price_before_normalization"),
        )
        .first()
        .asDict()
    )

    quality_metrics = silver_products.agg(
        F.count("*").alias("row_count"),
        F.sum(
            F.col("tenant_id").isNull().cast("long")
        ).alias("missing_tenant_id"),
        F.sum(
            F.col("product_id").isNull().cast("long")
        ).alias("missing_product_id"),
        F.sum(
            F.col("title").isNull().cast("long")
        ).alias("missing_title"),
        F.sum(
            F.col("brand").isNull().cast("long")
        ).alias("missing_brand"),
        F.sum(
            F.col("price").isNull().cast("long")
        ).alias("missing_price"),
        F.sum(
            F.col("source_record_id").isNull().cast("long")
        ).alias("missing_source_record_id"),
        F.sum(
            F.col("updated_at").isNull().cast("long")
        ).alias("missing_updated_at"),
        F.sum(
            ((F.col("price").isNotNull()) & (F.col("price") <= 0)).cast("long")        
        ).alias("invalid_price"),
    )

    metrics = quality_metrics.first()

    assert metrics.row_count > 0
    assert metrics.missing_tenant_id == 0

    valid_products = silver_products.filter(F.col("title").isNotNull())
    invalid_products = silver_products.filter(F.col("title").isNull())
    invalid_products = invalid_products.withColumn(
        "error_reason",
        F.lit("missing_title")
    )

    invalid_count = invalid_products.count()
    valid_count = valid_products.count()
    assert valid_count > 0
    assert invalid_count + valid_count == metrics.row_count

    valid_metrics = valid_products.agg(
        F.sum(
            F.col("tenant_id").isNull().cast("long")
        ).alias("missing_tenant_id"),
        F.sum(
            F.col("product_id").isNull().cast("long")
        ).alias("missing_product_id"),
        F.sum(
            F.col("title").isNull().cast("long")
        ).alias("missing_title"),
        F.sum(
            F.col("source_record_id").isNull().cast("long")
        ).alias("missing_source_record_id"),
        F.sum(
            F.col("updated_at").isNull().cast("long")
        ).alias("missing_updated_at"),
        F.sum(
            ((F.col("price").isNotNull()) & (F.col("price") <= 0)).cast("long")
        ).alias("invalid_price"),
    ).first()
    validate_quality_metrics(valid_metrics.asDict())

    duplicate_product_ids = (
        valid_products
        .groupBy(
            "tenant_id",
            "product_id",
        )
        .agg(
            F.count("*").alias("product_count"),
        )
        .filter(F.col("product_count") > 1)
        .select("tenant_id", "product_id")
        .count()
    )
    assert duplicate_product_ids == 0

    products_quality_metrics = {
        "row_count_before_dedup": product_source_metrics[
            "row_count_before_dedup"
        ],
        "row_count_after_dedup": metrics.row_count,
        "published_row_count": valid_count,
        "quarantine_row_count": invalid_count,
        "duplicate_product_keys": duplicate_product_ids,
        "missing_title": metrics.missing_title,
        "missing_brand": metrics.missing_brand,
        "missing_price": metrics.missing_price,
        "invalid_price": product_source_metrics[
            "invalid_price_before_normalization"
        ],
        "missing_product_id": metrics.missing_product_id,
    }
    # endregion: products

    # region: joins
    product_ids = valid_products.select("tenant_id", "product_id")
    orphan_reviews = silver_reviews.join(
        product_ids,
        on=["tenant_id", "product_id"],
        how="left_anti",
    )
    matched_reviews = silver_reviews.join(
        product_ids,
        on=["tenant_id", "product_id"],
        how="left_semi",
    )
    silver_reviews_count = row_count_after_dedup
    orphan_reviews_count = orphan_reviews.count()
    matched_reviews_count = matched_reviews.count()

    assert silver_reviews_count == orphan_reviews_count + matched_reviews_count

    reviews_quality_metrics = {
        "row_count_before_dedup": review_dedup_metrics[
            "row_count_before_dedup"
        ],
        "row_count_after_dedup": review_dedup_metrics[
            "row_count_after_dedup"
        ],
        "rows_removed_by_dedup": review_dedup_metrics[
            "rows_removed_by_dedup"
        ],
        "duplicate_review_keys": review_dedup_metrics[
            "duplicate_review_keys"
        ],
        "missing_product_id": review_violation_metrics[
            "missing_product_id"
        ],
        "missing_user_id": review_violation_metrics[
            "missing_user_id"
        ],
        "missing_event_time": review_violation_metrics[
            "missing_event_time"
        ],
        "invalid_rating": review_violation_metrics[
            "invalid_rating"
        ],
        "negative_helpful_vote": review_violation_metrics[
            "negative_helpful_vote"
        ],
        "orphan_review_count": orphan_reviews_count,
    }

    # endregion: joins

    review_evidence = select_review_evidence(
        matched_reviews,
        limit=20,
    )
    orphan_reviews = orphan_reviews.withColumn(
        "error_reason",
        F.lit("product_not_publishable"),
    )

    evidence_summary = (
        review_evidence
        .agg(
            F.count("*").alias("evidence_row_count"),
            F.max("evidence_rank").alias("max_evidence_rank"),
        )
        .first()
        .asDict()
    )

    evidence_group_sizes = (
        review_evidence
        .groupBy(
            "tenant_id",
            "product_id",
        )
        .agg(
            F.count("*").alias("evidence_count")
        )
    )

    max_evidence_per_product = (
        evidence_group_sizes
        .agg(
            F.max("evidence_count").alias(
                "max_evidence_per_product"
            )
        )
        .first()["max_evidence_per_product"]
    )

    evidence_metrics = {
        **evidence_summary,
        "max_evidence_per_product": max_evidence_per_product,
    }

    assert evidence_metrics["evidence_row_count"] > 0
    assert evidence_metrics["max_evidence_rank"] <= 20
    assert evidence_metrics["max_evidence_per_product"] <= 20

    print("Products quality metrics:", products_quality_metrics)
    print("Reviews quality metrics:", reviews_quality_metrics)
    print("Evidence quality metrics:", evidence_metrics)

    quality_metrics_rows = []
    quality_metrics_rows.extend(
        metrics_to_rows("products", run_date, products_quality_metrics)
    )
    quality_metrics_rows.extend(
        metrics_to_rows("reviews", run_date, reviews_quality_metrics)
    )
    quality_metrics_rows.extend(
        metrics_to_rows("review_evidence", run_date, evidence_metrics)
    )
    expected_quality_metrics_count = 23
    assert len(quality_metrics_rows) == expected_quality_metrics_count
    quality_metrics_df = spark.createDataFrame(
        quality_metrics_rows,
        ["entity", "run_date", "metric_name", "metric_value"],
    )

    evidence_for_join = review_evidence.withColumnRenamed("title", "review_title")
    product_lookup = valid_products.select(
        "tenant_id", 
        "product_id", 
        F.col("title").alias("product_title"),
        "brand",
        "category"
    )

    enriched_evidence = evidence_for_join.join(
        product_lookup, 
        on=["tenant_id", "product_id"], 
        how="left"
    )

    # region: dataset signatures
    valid_products_signature = calculate_dataset_signature(
        valid_products,
        ["tenant_id", "product_id"],
        ["tenant_id", "product_id"],
    )

    reviews_signature = calculate_dataset_signature(
        silver_reviews,
        ["tenant_id", "user_id", "product_id", "event_time"],
        ["review_id"],
    )

    evidence_signature = calculate_dataset_signature(
        enriched_evidence,
        ["tenant_id", "product_id", "evidence_rank"],
        ["tenant_id", "product_id", "evidence_rank", "review_id"],
    )

    current_signatures = {
        "products": valid_products_signature,
        "reviews": reviews_signature,
        "review_evidence": evidence_signature,
    }

    print("Current dataset signatures:", current_signatures)

    output_paths = [
        products_path,
        reviews_path,
        evidence_path,
    ]

    path_states = [
        Path(path).exists()
        for path in output_paths
    ]

    if any(path_states) and not all(path_states):
        raise RuntimeError(
            "Previous run is incomplete: only some output paths exist"
        )

    previous_outputs_exist = all(path_states)
    if previous_outputs_exist:
        previous_products = spark.read.parquet(products_path)
        previous_reviews = spark.read.parquet(reviews_path)
        previous_evidence = spark.read.parquet(evidence_path)
        previous_products_signature = calculate_dataset_signature(
            previous_products,
            ["tenant_id", "product_id"],
            ["tenant_id", "product_id"],
        )
        previous_reviews_signature = calculate_dataset_signature(
            previous_reviews,
            ["tenant_id", "user_id", "product_id", "event_time"],
            ["review_id"],
        )
        previous_evidence_signature = calculate_dataset_signature(
            previous_evidence,
            ["tenant_id", "product_id", "evidence_rank"],
            ["tenant_id", "product_id", "evidence_rank", "review_id"],
        )
        previous_signatures = {
            "products": previous_products_signature,
            "reviews": previous_reviews_signature,
            "review_evidence": previous_evidence_signature,
        }
        print("Previous dataset signatures:", previous_signatures)
        assert current_signatures == previous_signatures
    else:
        print("Previous outputs not found; rerun comparison skipped")

    # endregion: dataset signatures

    # region: data writing
    (
        valid_products.write
        .mode("overwrite")
        .partitionBy("tenant_id")
        .parquet(products_path)
    )

    (
        silver_reviews.write
        .mode("overwrite")
        .partitionBy("tenant_id")
        .parquet(reviews_path)
    )

    (
        enriched_evidence.write
        .mode("overwrite")
        .partitionBy("tenant_id")
        .parquet(evidence_path)
    )

    (
        invalid_products.write
        .mode("overwrite")
        .partitionBy("tenant_id")
        .parquet(product_quarantine_path)
    )

    (
        orphan_reviews.write
        .mode("overwrite")
        .partitionBy("tenant_id")
        .parquet(review_orphans_path)
    )

    (
        quality_metrics_df.write
        .mode("overwrite")
        .parquet(quality_metrics_path)
    )
    # endregion: data writing

    # region: data validation
    saved_products = spark.read.parquet(products_path)
    saved_reviews = spark.read.parquet(reviews_path)
    saved_evidence = spark.read.parquet(evidence_path)
    saved_quality_metrics = spark.read.parquet(quality_metrics_path)

    validate_required_columns(
        saved_products,
        "products",
        PRODUCT_REQUIRED_COLUMNS,
    )

    validate_required_columns(
        saved_reviews,
        "reviews",
        REVIEW_REQUIRED_COLUMNS,
    )

    saved_counts = {
        "products": saved_products.count(),
        "reviews": saved_reviews.count(),
        "review_evidence": saved_evidence.count(),
    }

    print("Saved counts:", saved_counts)

    expected_counts = {
        "products": valid_count,
        "reviews": row_count_after_dedup,
        "review_evidence": evidence_metrics["evidence_row_count"],
    }

    assert saved_counts == expected_counts
    assert (
        saved_quality_metrics.count()
        == expected_quality_metrics_count
    )

    saved_quarantine_counts = {
        "invalid_products": (
            spark.read.parquet(product_quarantine_path).count()
        ),
        "orphan_reviews": (
            spark.read.parquet(review_orphans_path).count()
        ),
    }

    print("Saved quarantine counts:", saved_quarantine_counts)

    assert saved_quarantine_counts == {
        "invalid_products": invalid_count,
        "orphan_reviews": orphan_reviews_count,
    }

    # endregion: data validation

    spark.stop()


if __name__ == "__main__":
    main()
