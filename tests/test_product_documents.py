from datetime import datetime

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from shoppilot.data.gold.document_quality import (
    calculate_product_document_metrics,
    validate_product_document_metrics,
)
from shoppilot.data.gold.documents import build_product_documents


PRODUCT_SCHEMA = """
tenant_id string,
product_id string,
title string,
category string,
brand string,
price double,
features array<string>,
description array<string>,
details map<string,string>,
updated_at timestamp,
is_active boolean
"""


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("ProductDocumentTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def product_row(
    product_id="p-1",
    *,
    tenant_id="beauty-demo",
    title="Face Cream",
    category="Skin Care",
    brand="Example Brand",
    price=19.99,
    features=None,
    description=None,
    details=None,
    is_active=True,
):
    return (
        tenant_id,
        product_id,
        title,
        category,
        brand,
        price,
        ["Hydrating", "Fragrance Free"] if features is None else features,
        ["Daily moisturizer", "For dry skin"] if description is None else description,
        {"Size": "50 ml", "Color": "White"} if details is None else details,
        datetime(2023, 1, 1),
        is_active,
    )


def test_product_document_has_canonical_section_order(spark):
    products = spark.createDataFrame([product_row()], PRODUCT_SCHEMA)

    text = build_product_documents(products).first()["document_text"]

    assert text == "\n".join(
        [
            "title: Face Cream",
            "category: Skin Care",
            "brand: Example Brand",
            "features: Hydrating | Fragrance Free",
            "description: Daily moisturizer | For dry skin",
            "details: Color=White | Size=50 ml",
        ]
    )


def test_product_document_skips_null_and_empty_sections(spark):
    products = spark.createDataFrame(
        [
            product_row(
                "p-minimal",
                title="Minimal Product",
                category=None,
                brand=None,
                features=[],
                description=[],
                details={},
            )
        ],
        PRODUCT_SCHEMA,
    )

    assert build_product_documents(products).first()["document_text"] == (
        "title: Minimal Product"
    )


def test_product_documents_exclude_inactive_products(spark):
    products = spark.createDataFrame(
        [product_row("active"), product_row("inactive", is_active=False)],
        PRODUCT_SCHEMA,
    )

    ids = [
        row["product_id"]
        for row in build_product_documents(products).select("product_id").collect()
    ]

    assert ids == ["active"]


def test_document_id_is_stable_and_tenant_scoped(spark):
    products = spark.createDataFrame(
        [
            product_row("shared", tenant_id="tenant-a"),
            product_row("shared", tenant_id="tenant-b"),
        ],
        PRODUCT_SCHEMA,
    )
    original = build_product_documents(products, document_version="v-test")
    changed = build_product_documents(
        products.withColumn("title", F.concat("title", F.lit(" Updated"))),
        document_version="v-test",
    )

    original_ids = {
        row["tenant_id"]: row["document_id"]
        for row in original.select("tenant_id", "document_id").collect()
    }
    changed_ids = {
        row["tenant_id"]: row["document_id"]
        for row in changed.select("tenant_id", "document_id").collect()
    }

    assert original_ids == changed_ids
    assert original_ids["tenant-a"] != original_ids["tenant-b"]
    assert original.select("document_version").first()[0] == "v-test"


def test_content_hash_normalizes_map_order_and_tracks_content(spark):
    parts = spark.createDataFrame(
        [
            ("ordered", "Face Cream", ["Size", "Color"], ["50 ml", "White"]),
            ("reordered", "Face Cream", ["Color", "Size"], ["White", "50 ml"]),
            ("changed", "Updated Cream", ["Size", "Color"], ["50 ml", "White"]),
        ],
        "product_id string, title string, keys array<string>, values array<string>",
    )
    products = parts.select(
        F.lit("beauty-demo").alias("tenant_id"),
        "product_id",
        "title",
        F.lit("Skin Care").alias("category"),
        F.lit("Example Brand").alias("brand"),
        F.lit(19.99).alias("price"),
        F.array(F.lit("Hydrating")).alias("features"),
        F.array(F.lit("Daily moisturizer")).alias("description"),
        F.map_from_arrays("keys", "values").alias("details"),
        F.lit(datetime(2023, 1, 1)).cast("timestamp").alias("updated_at"),
        F.lit(True).alias("is_active"),
    )
    hashes = {
        row["product_id"]: row["content_hash"]
        for row in build_product_documents(products)
        .select("product_id", "content_hash")
        .collect()
    }

    assert hashes["ordered"] == hashes["reordered"]
    assert hashes["ordered"] != hashes["changed"]


def test_price_remains_structured_and_is_not_embedded_in_text(spark):
    document = build_product_documents(
        spark.createDataFrame([product_row()], PRODUCT_SCHEMA)
    ).first()

    assert document["price"] == pytest.approx(19.99)
    assert "19.99" not in document["document_text"]
    assert "price:" not in document["document_text"]


def test_product_document_quality_gate(spark):
    documents = spark.createDataFrame(
        [
            ("doc-1", "title: First", "hash-1", True),
            ("doc-1", "", "hash-2", True),
            (None, None, None, True),
            ("doc-3", "title: Inactive", None, False),
        ],
        "document_id string, document_text string, content_hash string, is_active boolean",
    )

    metrics = calculate_product_document_metrics(documents)

    assert metrics == {
        "product_document_count": 4,
        "duplicate_document_id_count": 1,
        "missing_document_id_count": 1,
        "empty_document_text_count": 2,
        "missing_content_hash_count": 2,
        "inactive_document_count": 1,
    }
    with pytest.raises(ValueError, match="duplicate_document_id_count: 1"):
        validate_product_document_metrics(metrics)
