"""Readiness checks at the Silver-to-Gold boundary."""

from pyspark.sql import functions as F


PRODUCT_REQUIRED_COLUMNS = (
    "tenant_id", "product_id", "title", "category", "brand", "price",
    "features", "description", "details", "updated_at", "is_active",
)
REVIEW_REQUIRED_COLUMNS = (
    "tenant_id", "review_id", "product_id", "user_id", "rating",
    "helpful_vote", "event_time",
)


def validate_required_columns(dataframe, entity: str, required_columns) -> None:
    missing = sorted(set(required_columns) - set(dataframe.columns))
    if missing:
        raise ValueError(f"{entity} is missing required columns: {', '.join(missing)}")


def _sum_condition(condition, alias):
    return F.coalesce(F.sum(condition.cast("long")), F.lit(0)).alias(alias)


def validate_silver_readiness(products_df, reviews_df) -> dict:
    validate_required_columns(products_df, "silver products", PRODUCT_REQUIRED_COLUMNS)
    validate_required_columns(reviews_df, "silver reviews", REVIEW_REQUIRED_COLUMNS)

    product_row = products_df.agg(
        F.count("*").alias("silver_product_count"),
        F.countDistinct(F.struct("tenant_id", "product_id")).alias(
            "distinct_product_key_count"
        ),
        _sum_condition(F.col("is_active") == F.lit(True), "active_product_count"),
    ).first()
    missing_review_key = (
        F.col("tenant_id").isNull()
        | F.col("review_id").isNull()
        | F.col("product_id").isNull()
        | F.col("user_id").isNull()
        | F.col("rating").isNull()
        | F.col("event_time").isNull()
    )
    review_row = reviews_df.agg(
        F.count("*").alias("silver_review_count"),
        F.countDistinct(F.struct("tenant_id", "review_id")).alias(
            "distinct_review_key_count"
        ),
        _sum_condition(missing_review_key, "missing_required_review_value_count"),
        _sum_condition(
            F.col("helpful_vote").isNull() | (F.col("helpful_vote") < 0),
            "invalid_helpful_vote_count",
        ),
        _sum_condition(
            (F.col("rating") < 1) | (F.col("rating") > 5),
            "invalid_rating_count",
        ),
    ).first()
    metrics = {
        "silver_product_count": product_row["silver_product_count"],
        "active_product_count": product_row["active_product_count"],
        "duplicate_product_key_count": (
            product_row["silver_product_count"] - product_row["distinct_product_key_count"]
        ),
        "silver_review_count": review_row["silver_review_count"],
        "duplicate_review_id_count": (
            review_row["silver_review_count"] - review_row["distinct_review_key_count"]
        ),
        "missing_required_review_value_count": review_row[
            "missing_required_review_value_count"
        ],
        "invalid_helpful_vote_count": review_row["invalid_helpful_vote_count"],
        "invalid_rating_count": review_row["invalid_rating_count"],
    }
    violation_names = (
        "duplicate_product_key_count", "duplicate_review_id_count",
        "missing_required_review_value_count", "invalid_helpful_vote_count",
        "invalid_rating_count",
    )
    violations = [
        f"{name}: {metrics[name]}" for name in violation_names if metrics[name] > 0
    ]
    if violations:
        raise ValueError("Silver readiness gate failed:\n" + "\n".join(violations))
    return metrics
