import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.testing.utils import assertDataFrameEqual

from shoppilot.jobs.silver import validate_quality_metrics
from shoppilot.data.silver.transforms import (
    deduplicate_products,
    deduplicate_reviews,
    generate_review_id,
    normalize_event_time,
    normalize_price,
    normalize_reviews,
    normalize_string,
    normalize_string_array,
    resolve_category,
    select_review_evidence,
)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder
        .master("local[2]")
        .appName("SilverTransformsTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_normalize_string(spark):
    source = spark.createDataFrame(
        [
            (1, None),
            (2, "   "),
            (3, "  Aloe   Vera  "),
            (4, " Aloe \t Vera \n Gel "),
        ],
        "id int, value string",
    )

    actual = (
        source
        .select(
            "id",
            normalize_string(F.col("value")).alias("normalized"),
        )
        .orderBy("id")
        .collect()
    )

    assert [row["normalized"] for row in actual] == [
        None,
        None,
        "Aloe Vera",
        "Aloe Vera Gel",
    ]


def test_normalize_string_array(spark):
    source = spark.createDataFrame(
        [
            (1, None),
            (2, []),
            (3, ["  Aloe  ", "", None, "Aloe", " Gel "]),
            (4, [" A   B ", "A B"]),
            (5, ["One", " Two ", "One", "  Three   Words  "]),
        ],
        "id int, values array<string>",
    )

    actual = (
        source
        .select(
            "id",
            normalize_string_array(F.col("values")).alias("normalized"),
        )
        .orderBy("id")
        .collect()
    )

    assert [row["normalized"] for row in actual] == [
        None,
        [],
        ["Aloe", "Gel"],
        ["A B"],
        ["One", "Two", "Three Words"],
    ]


def test_resolve_category(spark):
    source = spark.createDataFrame(
        [
            (1, ["  Skin Category  ", "", None, " Face "], "Beauty"),
            (2, ["   ", " Face "], "All Beauty"),
            (3, [], "  All  Beauty  "),
            (4, None, "  All  Beauty  "),
            (5, None, None),
        ],
        "id int, categories array<string>, main_category string",
    )

    actual = (
        source
        .select(
            "id",
            resolve_category(
                F.col("categories"),
                F.col("main_category"),
            ).alias("category"),
        )
        .orderBy("id")
        .collect()
    )

    assert [row["category"] for row in actual] == [
        "Skin Category",
        "Face",
        "All Beauty",
        "All Beauty",
        None,
    ]


def test_normalize_price(spark):
    source = spark.createDataFrame(
        [
            (1, None),
            (2, -10.0),
            (3, 0.0),
            (4, 0.01),
            (5, 100.0),
        ],
        "id int, price double",
    )

    actual = (
        source
        .select("id", normalize_price(F.col("price")).alias("price"))
        .orderBy("id")
        .collect()
    )

    assert [row["price"] for row in actual] == [
        None,
        None,
        None,
        0.01,
        100.0,
    ]


def test_product_dedup_follows_business_order(spark):
    source = spark.createDataFrame(
        [
            ("beauty-demo", "P1", 5, "2026-10-20 10:00:00", "b", "more_complete"),
            ("beauty-demo", "P1", 4, "2026-10-21 10:00:00", "a", "newer_less_complete"),
            ("beauty-demo", "P2", 5, "2026-10-20 10:00:00", "a", "older"),
            ("beauty-demo", "P2", 5, "2026-10-21 10:00:00", "b", "newer"),
            ("beauty-demo", "P3", 5, "2026-10-21 10:00:00", "b", "larger_id"),
            ("beauty-demo", "P3", 5, "2026-10-21 10:00:00", "a", "smaller_id"),
        ],
        """
        tenant_id string,
        product_id string,
        _completeness_score int,
        updated_at string,
        source_record_id string,
        version string
        """,
    ).withColumn("updated_at", F.to_timestamp("updated_at"))

    actual = {
        row["product_id"]: row["version"]
        for row in (
            deduplicate_products(source)
            .select("product_id", "version")
            .collect()
        )
    }

    assert actual == {
        "P1": "more_complete",
        "P2": "newer",
        "P3": "smaller_id",
    }


def test_orphan_detection_with_left_anti_join(spark):
    products = spark.createDataFrame(
        [
            ("tenant-a", "P1"),
            ("tenant-b", "P2"),
        ],
        "tenant_id string, product_id string",
    )
    reviews = spark.createDataFrame(
        [
            ("tenant-a", "R1", "P1"),
            ("tenant-a", "R2", "P2"),
            ("tenant-a", "R3", "P3"),
            ("tenant-b", "R4", "P2"),
        ],
        "tenant_id string, review_id string, product_id string",
    )
    expected = spark.createDataFrame(
        [
            ("tenant-a", "R2", "P2"),
            ("tenant-a", "R3", "P3"),
        ],
        "tenant_id string, review_id string, product_id string",
    ).orderBy("review_id")

    product_keys = products.select("tenant_id", "product_id").distinct()
    actual = (
        reviews
        .join(
            product_keys,
            on=["tenant_id", "product_id"],
            how="left_anti",
        )
        .select("tenant_id", "review_id", "product_id")
        .orderBy("review_id")
    )

    assertDataFrameEqual(actual, expected)


def test_review_timestamp_is_converted_from_milliseconds(spark):
    source = spark.createDataFrame(
        [
            ("regular", 1503577989475),
            ("millisecond_precision", 1503577989001),
            ("exact_second", 1503577989000),
            ("unix_epoch", 0),
            ("null_value", None),
        ],
        "case_name string, timestamp_ms long",
    )

    result = (
        source
        .withColumn(
            "event_time",
            normalize_event_time(F.col("timestamp_ms")),
        )
        .withColumn(
            "restored_millis",
            F.unix_millis(F.col("event_time")),
        )
        .withColumn(
            "is_equal",
            F.col("timestamp_ms").eqNullSafe(F.col("restored_millis")),
        )
    )

    assert result.filter(~F.col("is_equal")).count() == 0
    assert dict(result.dtypes)["event_time"] == "timestamp"


def test_review_null_semantics_are_explicit(spark):
    source = spark.createDataFrame(
        [
            (
                "beauty-demo",
                "P1",
                "U1",
                5.0,
                "   ",
                "  useful   text ",
                None,
                None,
                1503577989475,
                "source-1",
            ),
        ],
        """
        tenant_id string,
        parent_asin string,
        user_id string,
        rating double,
        title string,
        text string,
        verified_purchase boolean,
        helpful_vote long,
        timestamp long,
        source_record_id string
        """,
    )

    row = normalize_reviews(source).first()

    assert row["title"] is None
    assert row["text"] == "useful text"
    assert row["verified_purchase"] is None
    assert row["helpful_vote"] == 0


def test_review_id_is_stable(spark):
    source = spark.createDataFrame(
        [
            (1, "beauty-demo", "U1", "P1", 1503577989475, "a" * 64),
            (2, "beauty-demo", "U1", "P1", 1503577989475, "a" * 64),
            (3, "beauty-demo", "U1", "P1", 1503577989475, "b" * 64),
        ],
        """
        case_id int,
        tenant_id string,
        user_id string,
        product_id string,
        event_time_ms long,
        source_record_id string
        """,
    ).withColumn(
        "event_time",
        F.timestamp_millis(F.col("event_time_ms")),
    )

    rows = (
        source
        .withColumn("review_id", generate_review_id())
        .orderBy("case_id")
        .collect()
    )

    assert rows[0]["review_id"] == rows[1]["review_id"]
    assert rows[0]["review_id"] != rows[2]["review_id"]
    assert all(len(row["review_id"]) == 64 for row in rows)


def test_review_dedup_follows_business_order(spark):
    source = spark.createDataFrame(
        [
            ("verified_winner", "U1", True, 0, "short", "review-z"),
            ("verified_loser", "U1", False, 100, "a much longer text", "review-a"),
            ("helpful_winner", "U2", False, 10, "short", "review-z"),
            ("helpful_loser", "U2", False, 5, "a much longer text", "review-a"),
            ("text_winner", "U3", False, 3, "this text is longer", "review-z"),
            ("text_loser", "U3", False, 3, "short", "review-a"),
            ("id_winner", "U4", False, 3, "same text", "review-a"),
            ("id_loser", "U4", False, 3, "same text", "review-b"),
        ],
        """
        candidate string,
        user_id string,
        verified_purchase boolean,
        helpful_vote long,
        text string,
        review_id string
        """,
    ).select(
        "*",
        F.lit("beauty-demo").alias("tenant_id"),
        F.lit("P1").alias("product_id"),
        F.to_timestamp(F.lit("2026-10-20 10:00:00")).alias("event_time"),
    )

    result = deduplicate_reviews(source)
    actual = {
        row["user_id"]: row["candidate"]
        for row in result.select("user_id", "candidate").collect()
    }

    assert actual == {
        "U1": "verified_winner",
        "U2": "helpful_winner",
        "U3": "text_winner",
        "U4": "id_winner",
    }
    assert result.count() == 4
    assert "_row_number" not in result.columns
    assert "_verified_purchase_score" not in result.columns


def evidence_order_fixture(spark):
    return spark.createDataFrame(
        [
            ("P1", "p1-verified", True, 0, "2026-10-01 10:00:00"),
            ("P1", "p1-helpful", False, 10, "2026-10-01 10:00:00"),
            ("P1", "p1-newer", False, 5, "2026-10-05 10:00:00"),
            ("P1", "p1-id-a", False, 5, "2026-10-04 10:00:00"),
            ("P1", "p1-id-b", False, 5, "2026-10-04 10:00:00"),
            ("P2", "p2-first", False, 2, "2026-10-03 10:00:00"),
            ("P2", "p2-second", False, 1, "2026-10-03 10:00:00"),
        ],
        """
        product_id string,
        review_id string,
        verified_purchase boolean,
        helpful_vote long,
        event_time string
        """,
    ).select(
        "*",
        F.lit("beauty-demo").alias("tenant_id"),
    ).withColumn("event_time", F.to_timestamp("event_time"))


def collect_evidence_ranking(dataframe):
    return [
        (row["product_id"], row["review_id"], row["evidence_rank"])
        for row in (
            dataframe
            .orderBy("product_id", "evidence_rank")
            .select("product_id", "review_id", "evidence_rank")
            .collect()
        )
    ]


def test_evidence_order_and_partitioned_ranking(spark):
    result = select_review_evidence(
        evidence_order_fixture(spark),
        limit=3,
    )

    assert collect_evidence_ranking(result) == [
        ("P1", "p1-verified", 1),
        ("P1", "p1-helpful", 2),
        ("P1", "p1-newer", 3),
        ("P2", "p2-first", 1),
        ("P2", "p2-second", 2),
    ]


def test_evidence_limit_and_ties_are_deterministic(spark):
    source = spark.createDataFrame(
        [
            (
                "beauty-demo",
                "P-limit",
                f"review-{index:02d}",
                False,
                0,
                "2026-10-03 10:00:00",
            )
            for index in range(22)
        ],
        """
        tenant_id string,
        product_id string,
        review_id string,
        verified_purchase boolean,
        helpful_vote long,
        event_time string
        """,
    ).withColumn("event_time", F.to_timestamp("event_time"))

    first_result = select_review_evidence(source, limit=20)
    reordered_source = source.orderBy(F.desc("review_id"))
    second_result = select_review_evidence(reordered_source, limit=20)

    expected = [
        ("P-limit", f"review-{index:02d}", index + 1)
        for index in range(20)
    ]

    assert collect_evidence_ranking(first_result) == expected
    assert collect_evidence_ranking(second_result) == expected


def test_evidence_rejects_non_positive_limit(spark):
    with pytest.raises(ValueError, match="greater than 0"):
        select_review_evidence(evidence_order_fixture(spark), limit=0)


def test_validate_quality_metrics():
    validate_quality_metrics(
        {
            "missing_tenant_id": 0,
            "missing_product_id": 0,
            "missing_title": 0,
            "invalid_price": 0,
        }
    )

    with pytest.raises(ValueError) as error:
        validate_quality_metrics(
            {
                "missing_tenant_id": 0,
                "missing_product_id": 0,
                "missing_title": 2,
                "invalid_price": 1,
            }
        )

    message = str(error.value)
    assert "missing_title: 2" in message
    assert "invalid_price: 1" in message
