"""Deterministic transformations from Bronze to Silver."""

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# region products
def normalize_string(column):
    collapsed = F.regexp_replace(column, r"\s+", " ")
    trimmed = F.trim(collapsed)
    return F.when(trimmed == "", F.lit(None)).otherwise(trimmed)


def normalize_string_array(column):
    normalized_array = F.transform(
        column,
        lambda item: normalize_string(item),
    )
    non_empty_array = F.filter(
        normalized_array,
        lambda item: item.isNotNull(),
    )
    return F.array_distinct(non_empty_array)


def normalize_price(column):
    return F.when(column <= 0.0, F.lit(None)).otherwise(column)


def resolve_brand(brand_from_store, brand_from_details):
    return F.coalesce(brand_from_store, brand_from_details)


def resolve_category(categories, main_category):
    non_empty_categories = normalize_string_array(categories)
    first_category = F.try_element_at(non_empty_categories, F.lit(1))
    normalized_main_category = normalize_string(main_category)
    return F.coalesce(first_category, normalized_main_category)


def product_completeness_score():
    title_not_null = F.when(F.col("title").isNotNull(), 1).otherwise(0)
    category_not_null = F.when(F.col("category").isNotNull(), 1).otherwise(0)
    brand_not_null = F.when(F.col("brand").isNotNull(), 1).otherwise(0)
    price_not_null = F.when(F.col("price").isNotNull(), 1).otherwise(0)
    features_non_empty = F.when(F.size(F.col("features")) > 0, 1).otherwise(0)
    description_non_empty = F.when(
        F.size(F.col("description")) > 0,
        1,
    ).otherwise(0)
    details_non_empty = F.when(F.size(F.col("details")) > 0, 1).otherwise(0)
    return (
        title_not_null
        + category_not_null
        + brand_not_null
        + price_not_null
        + features_non_empty
        + description_non_empty
        + details_non_empty
    )


def normalize_products(metadata_df):
    brand_from_store = normalize_string(F.col("store"))
    brand_from_details = normalize_string(
        F.element_at(F.col("details"), "Brand")
    )
    brand = resolve_brand(brand_from_store, brand_from_details)
    category = resolve_category(F.col("categories"), F.col("main_category"))
    return metadata_df.select(
        F.col("parent_asin").alias("product_id"),
        F.col("ingested_at").alias("updated_at"),
        category.alias("category"),
        "details",
        brand.alias("brand"),
        normalize_string(F.col("title")).alias("title"),
        normalize_price(F.col("price")).alias("price"),
        normalize_string_array(F.col("features")).alias("features"),
        normalize_string_array(F.col("description")).alias("description"),
        "tenant_id",
        "source_record_id",
        F.lit(True).alias("is_active"),
    )


def deduplicate_products(products_df):
    window = Window.partitionBy("tenant_id", "product_id").orderBy(
        F.desc("_completeness_score"),
        F.desc("updated_at"),
        F.asc("source_record_id"),
    )
    return (
        products_df
        .withColumn("_row_number", F.row_number().over(window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "_completeness_score")
    )


def build_silver_products(metadata_df):
    normalized_products = normalize_products(metadata_df)
    scored_products = normalized_products.withColumn(
        "_completeness_score",
        product_completeness_score(),
    )
    return deduplicate_products(scored_products)


# endregion products

# region reviews
def normalize_event_time(column):
    return F.timestamp_millis(column)


def normalize_helpful_vote(column):
    return F.coalesce(column, F.lit(0))


def normalize_reviews(reviews_df):
    return reviews_df.select(
        "tenant_id",
        F.col("parent_asin").alias("product_id"),
        "user_id",
        "rating",
        normalize_string(F.col("title")).alias("title"),
        normalize_string(F.col("text")).alias("text"),
        "verified_purchase",
        normalize_helpful_vote(F.col("helpful_vote")).alias("helpful_vote"),
        normalize_event_time(F.col("timestamp")).alias("event_time"),
        "source_record_id",
    )


def generate_review_id():
    identity_fields = F.struct(
        F.col("tenant_id"),
        F.col("user_id"),
        F.col("product_id"),
        F.unix_millis(F.col("event_time")).alias("event_time_ms"),
        F.col("source_record_id"),
    )
    canonical_json = F.to_json(
        identity_fields,
        options={"ignoreNullFields": "false"},
    )
    return F.sha2(canonical_json, 256)


def deduplicate_reviews(reviews_df):
    reviews_with_score = reviews_df.withColumn(
        "_verified_purchase_score",
        F.when(F.col("verified_purchase") == F.lit(True), 1).otherwise(0),
    )
    window = Window.partitionBy(
        "tenant_id",
        "user_id",
        "product_id",
        "event_time",
    ).orderBy(
        F.desc("_verified_purchase_score"),
        F.desc("helpful_vote"),
        F.length(F.col("text")).desc_nulls_last(),
        F.asc("review_id"),
    )
    return (
        reviews_with_score
        .withColumn("_row_number", F.row_number().over(window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "_verified_purchase_score")
    )


def build_silver_reviews(reviews_df):
    normalized_reviews = normalize_reviews(reviews_df)
    reviews_with_id = normalized_reviews.withColumn(
        "review_id",
        generate_review_id(),
    )
    deduplicated_reviews = deduplicate_reviews(reviews_with_id)
    return deduplicated_reviews.select(
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
        "source_record_id",
    )


def select_review_evidence(reviews_df, limit: int = 20):
    if limit <= 0:
        raise ValueError("Limit must be greater than 0")

    window = Window.partitionBy("tenant_id", "product_id").orderBy(
        F.desc(
            F.when(
                F.col("verified_purchase") == F.lit(True),
                1,
            ).otherwise(0)
        ),
        F.desc("helpful_vote"),
        F.desc("event_time"),
        F.asc("review_id"),
    )

    return (
        reviews_df
        .withColumn("evidence_rank", F.row_number().over(window))
        .filter(F.col("evidence_rank") <= F.lit(limit))
    )


# endregion reviews
