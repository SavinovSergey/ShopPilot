import pytest
from pyspark.sql import SparkSession

from shoppilot.data.bronze.transforms import (
    add_ingestion_metadata,
    source_record_id_expr,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("BronzeTransformTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_source_record_id_is_stable_for_map_key_order(spark):
    dataframe = spark.createDataFrame(
        [("p1", {"b": "2", "a": "1"})],
        "product_id string, details map<string,string>",
    )
    reordered = spark.createDataFrame(
        [("p1", {"a": "1", "b": "2"})],
        dataframe.schema,
    )

    left_id = dataframe.select(
        source_record_id_expr(dataframe, "metadata").alias("id")
    ).first()["id"]
    right_id = reordered.select(
        source_record_id_expr(reordered, "metadata").alias("id")
    ).first()["id"]

    assert left_id == right_id


def test_ingestion_metadata_contains_rebuild_keys(spark):
    source = spark.createDataFrame(
        [(5.0, "p1", "u1")],
        "rating double, parent_asin string, user_id string",
    )

    row = add_ingestion_metadata(
        source,
        tenant_id="beauty-demo",
        batch_id="batch-1",
        ingestion_date="2026-10-21",
        source_kind="reviews",
    ).first()

    assert row["tenant_id"] == "beauty-demo"
    assert row["batch_id"] == "batch-1"
    assert row["ingestion_date"].isoformat() == "2026-10-21"
    assert row["source_record_id"] is not None
    assert row["schema_version"] == 1
