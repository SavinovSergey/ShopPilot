"""Explicit source and Bronze schemas for Amazon Reviews 2023."""

from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    MapType,
    TimestampType,
)


def schema_signature(schema):
    return {
        field.name: field.dataType.simpleString()
        for field in schema.fields
    }


review_image_schema = StructType([
    StructField("small_image_url", StringType(), True),
    StructField("medium_image_url", StringType(), True),
    StructField("large_image_url", StringType(), True),
    StructField("attachment_type", StringType(), True),
])

reviews_source_schema = StructType([
    StructField("rating", DoubleType(), True),
    StructField("title", StringType(), True),
    StructField("text", StringType(), True),
    StructField("images", ArrayType(review_image_schema), True),
    StructField("asin", StringType(), True),
    StructField("parent_asin", StringType(), True),
    StructField("user_id", StringType(), True),
    StructField("timestamp", LongType(), True),
    StructField("helpful_vote", LongType(), True),
    StructField("verified_purchase", BooleanType(), True),
])

reviews_reader_schema = StructType(
    reviews_source_schema.fields + [
        StructField("_corrupt_record", StringType(), True),
    ]
)

metadata_image_schema = StructType([
    StructField("thumb", StringType(), True),
    StructField("large", StringType(), True),
    StructField("variant", StringType(), True),
    StructField("hi_res", StringType(), True),
])

metadata_video_schema = StructType([
    StructField("title", StringType(), True),
    StructField("url", StringType(), True),
    StructField("user_id", StringType(), True),
])

metadata_schema = StructType([
    StructField("main_category", StringType(), True),
    StructField("title", StringType(), True),
    StructField("average_rating", DoubleType(), True),
    StructField("price", DoubleType(), True),
    StructField("rating_number", LongType(), True),
    StructField("features", ArrayType(StringType()), True),
    StructField("description", ArrayType(StringType()), True),
    StructField("images", ArrayType(metadata_image_schema), True),
    StructField("videos", ArrayType(metadata_video_schema), True),
    StructField("store", StringType(), True),
    StructField("categories", ArrayType(StringType()), True),
    StructField("details", MapType(StringType(), StringType()), True),
    StructField("parent_asin", StringType(), True),
    StructField("bought_together", ArrayType(StringType()), True),
    StructField("_corrupt_record", StringType(), True),
])


# region: Bronze Schemas
reviews_bronze_schema = StructType([
    *reviews_source_schema.fields,
    StructField("source_file", StringType(), True),
    StructField("ingested_at", TimestampType(), False),
    StructField("batch_id", StringType(), False),
])

metadata_bronze_schema = StructType([
    *metadata_schema.fields,
    StructField("source_file", StringType(), True),
    StructField("ingested_at", TimestampType(), False),
    StructField("batch_id", StringType(), False),
])
# endregion
