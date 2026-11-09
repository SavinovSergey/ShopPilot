"""Quality checks for retrieval product documents."""

from pyspark.sql import functions as F


def calculate_product_document_metrics(documents_df) -> dict:
    row = documents_df.agg(
        F.count("*").alias("product_document_count"),
        F.count("document_id").alias("non_null_document_id_count"),
        F.countDistinct("document_id").alias("distinct_document_id_count"),
        F.coalesce(F.sum((F.col("document_id").isNull() | (F.trim("document_id") == "")).cast("long")), F.lit(0)).alias("missing_document_id_count"),
        F.coalesce(F.sum((F.col("document_text").isNull() | (F.trim("document_text") == "")).cast("long")), F.lit(0)).alias("empty_document_text_count"),
        F.coalesce(F.sum((F.col("content_hash").isNull() | (F.trim("content_hash") == "")).cast("long")), F.lit(0)).alias("missing_content_hash_count"),
        F.coalesce(F.sum((~F.coalesce(F.col("is_active"), F.lit(False))).cast("long")), F.lit(0)).alias("inactive_document_count"),
    ).first()
    return {
        "product_document_count": row["product_document_count"],
        "duplicate_document_id_count": row["non_null_document_id_count"] - row["distinct_document_id_count"],
        "missing_document_id_count": row["missing_document_id_count"],
        "empty_document_text_count": row["empty_document_text_count"],
        "missing_content_hash_count": row["missing_content_hash_count"],
        "inactive_document_count": row["inactive_document_count"],
    }


def validate_product_document_metrics(metrics: dict) -> None:
    names = (
        "duplicate_document_id_count", "missing_document_id_count",
        "empty_document_text_count", "missing_content_hash_count",
        "inactive_document_count",
    )
    errors = [f"{name}: {metrics[name]}" for name in names if metrics[name] > 0]
    if errors:
        raise ValueError("Invalid product document metrics:\n" + "\n".join(errors))
