"""Build canonical product documents for retrieval and embeddings."""

from pyspark.sql import functions as F


PRODUCT_DOCUMENT_OUTPUT_COLUMNS = (
    "tenant_id",
    "product_id",
    "document_id",
    "document_text",
    "content_hash",
    "title",
    "category",
    "brand",
    "price",
    "features",
    "description",
    "details",
    "is_active",
    "updated_at",
    "document_version",
)


def build_product_documents(products_df, document_version="v1"):
    active_products = products_df.filter(F.col("is_active") == F.lit(True))
    sorted_details = F.array_sort(F.map_entries(F.col("details")))
    detail_lines = F.transform(
        sorted_details,
        lambda item: F.concat_ws("=", item["key"], item["value"]),
    )

    document_text = F.concat_ws(
        "\n",
        F.when(
            F.col("title").isNotNull(),
            F.concat_ws(": ", F.lit("title"), F.col("title")),
        ),
        F.when(
            F.col("category").isNotNull(),
            F.concat_ws(": ", F.lit("category"), F.col("category")),
        ),
        F.when(
            F.col("brand").isNotNull(),
            F.concat_ws(": ", F.lit("brand"), F.col("brand")),
        ),
        F.when(
            F.size(F.col("features")) > 0,
            F.concat_ws(
                ": ",
                F.lit("features"),
                F.concat_ws(" | ", F.col("features")),
            ),
        ),
        F.when(
            F.size(F.col("description")) > 0,
            F.concat_ws(
                ": ",
                F.lit("description"),
                F.concat_ws(" | ", F.col("description")),
            ),
        ),
        F.when(
            F.size(sorted_details) > 0,
            F.concat_ws(
                ": ",
                F.lit("details"),
                F.concat_ws(" | ", detail_lines),
            ),
        ),
    )

    document_id = F.sha2(
        F.to_json(F.struct(F.col("tenant_id"), F.col("product_id"))),
        256,
    )
    canonical_content = F.to_json(
        F.struct(
            F.col("title").alias("title"),
            F.col("category").alias("category"),
            F.col("brand").alias("brand"),
            F.col("features").alias("features"),
            F.col("description").alias("description"),
            detail_lines.alias("details"),
        ),
    )

    return (
        active_products.withColumn("document_text", document_text)
        .withColumn("document_id", document_id)
        .withColumn("document_version", F.lit(document_version))
        .withColumn("content_hash", F.sha2(canonical_content, 256))
    )
