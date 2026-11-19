import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.popularity import (
    rank_global_popularity,
    recommend_global_popularity,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderPopularityTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_global_popularity_ranks_per_tenant_and_breaks_ties(spark):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    item_popularity = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-b", 5),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-c", 2),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-a", 5),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-2", "product-z", 10),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "wrong-snapshot", 999),
            ("snapshot-v1", "signal-v2", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "wrong-signal", 999),
            ("snapshot-v1", "signal-v1", "protocol-v2", "binary_unique_pair_v1", "tenant-1", "wrong-protocol", 999),
            ("snapshot-v1", "signal-v1", "protocol-v1", "other-aggregation", "tenant-1", "wrong-aggregation", 999),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        product_id string,
        positive_pair_count long
        """,
    )

    ranked = rank_global_popularity(item_popularity, config)

    assert {
        (row["tenant_id"], row["product_id"]): (
            row["positive_pair_count"], row["popularity_rank"]
        )
        for row in ranked.collect()
    } == {
        ("tenant-1", "product-a"): (5, 1),
        ("tenant-1", "product-b"): (5, 2),
        ("tenant-1", "product-c"): (2, 3),
        ("tenant-2", "product-z"): (10, 1),
    }


def test_category_popularity_ranks_train_items_within_tenant_and_category(spark):
    from shoppilot.recommendations.popularity import rank_category_popularity

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    item_popularity = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-b", 5),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-a", 5),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-c", 2),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-d", 7),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-null", 8),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-2", "product-a", 3),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "wrong-snapshot", 100),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        product_id string,
        positive_pair_count long
        """,
    )
    products = spark.createDataFrame(
        [
            ("tenant-1", "product-a", "skin"),
            ("tenant-1", "product-b", "skin"),
            ("tenant-1", "product-c", "skin"),
            ("tenant-1", "product-d", "hair"),
            ("tenant-1", "product-null", None),
            ("tenant-1", "catalog-only", "skin"),
            ("tenant-1", "wrong-snapshot", "skin"),
            ("tenant-2", "product-a", "hair"),
        ],
        "tenant_id string, product_id string, category string",
    )

    ranked = rank_category_popularity(item_popularity, products, config)

    assert {
        (row["tenant_id"], row["category"], row["product_id"]): (
            row["positive_pair_count"], row["category_rank"]
        )
        for row in ranked.collect()
    } == {
        ("tenant-1", "skin", "product-a"): (5, 1),
        ("tenant-1", "skin", "product-b"): (5, 2),
        ("tenant-1", "skin", "product-c"): (2, 3),
        ("tenant-1", "hair", "product-d"): (7, 1),
        ("tenant-2", "hair", "product-a"): (3, 1),
    }


def test_user_category_preferences_count_train_seen_items_per_category(spark):
    from shoppilot.recommendations.popularity import build_user_category_preferences

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-b"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-c"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-null"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "orphan"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-2", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-2", "product-c"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-2", "user-1", "product-a"),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-other-snapshot"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )
    products = spark.createDataFrame(
        [
            ("tenant-1", "product-a", "skin"),
            ("tenant-1", "product-b", "skin"),
            ("tenant-1", "product-c", "hair"),
            ("tenant-1", "product-null", None),
            ("tenant-1", "product-other-snapshot", "skin"),
            ("tenant-2", "product-a", "hair"),
        ],
        "tenant_id string, product_id string, category string",
    )

    preferences = build_user_category_preferences(seen_items, products, config)

    assert {
        (row["tenant_id"], row["user_id"], row["category"]): (
            row["seen_category_count"], row["category_preference_rank"]
        )
        for row in preferences.collect()
    } == {
        ("tenant-1", "user-1", "skin"): (2, 1),
        ("tenant-1", "user-1", "hair"): (1, 2),
        ("tenant-1", "user-2", "hair"): (1, 1),
        ("tenant-1", "user-2", "skin"): (1, 2),
        ("tenant-2", "user-1", "hair"): (1, 1),
    }


def test_category_candidates_join_preferences_and_exclude_versioned_seen(spark):
    from shoppilot.recommendations.popularity import build_user_category_candidates

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    preferences = spark.createDataFrame(
        [
            ("tenant-1", "user-1", "skin", 1, 1),
            ("tenant-1", "user-1", "hair", 1, 2),
            ("tenant-1", "user-2", "hair", 1, 1),
            ("tenant-2", "user-1", "skin", 1, 1),
        ],
        "tenant_id string, user_id string, category string, seen_category_count long, category_preference_rank int",
    )
    ranked_categories = spark.createDataFrame(
        [
            ("tenant-1", "skin", "product-a", 1),
            ("tenant-1", "skin", "product-b", 2),
            ("tenant-1", "hair", "product-c", 1),
            ("tenant-1", "hair", "product-d", 2),
            ("tenant-2", "skin", "product-a", 1),
        ],
        "tenant_id string, category string, product_id string, category_rank int",
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-c"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-2", "product-d"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-2", "user-1", "product-a"),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-b"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )

    candidates = build_user_category_candidates(
        preferences, ranked_categories, seen_items, config, k=2
    )

    assert {
        (row["tenant_id"], row["user_id"], row["product_id"]): (
            row["category"], row["category_preference_rank"], row["category_rank"]
        )
        for row in candidates.collect()
    } == {
        ("tenant-1", "user-1", "product-b"): ("skin", 1, 2),
        ("tenant-1", "user-1", "product-d"): ("hair", 2, 2),
        ("tenant-1", "user-2", "product-c"): ("hair", 1, 1),
    }
    assert candidates.count() == 3


def test_category_candidate_limit_preserves_top_k_after_seen_exclusion(spark):
    from shoppilot.recommendations.popularity import (
        build_user_category_candidates,
        recommend_from_category_candidates,
    )

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    users = spark.createDataFrame(
        [("tenant-1", "user-1"), ("tenant-1", "user-2")],
        "tenant_id string, user_id string",
    )
    preferences = spark.createDataFrame(
        [
            ("tenant-1", "user-1", "skin", 2, 1),
            ("tenant-1", "user-2", "skin", 1, 1),
        ],
        "tenant_id string, user_id string, category string, seen_category_count long, category_preference_rank int",
    )
    ranked_categories = spark.createDataFrame(
        [("tenant-1", "skin", product_id, rank) for rank, product_id in enumerate("abcdef", 1)],
        "tenant_id string, category string, product_id string, category_rank int",
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "b"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-2", "a"),
        ],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, user_id string,
        product_id string
        """,
    )

    candidates = build_user_category_candidates(
        preferences, ranked_categories, seen_items, config, k=2
    )
    recommendations = recommend_from_category_candidates(users, candidates, k=2)

    assert max(row["category_rank"] for row in candidates.collect()) <= 4
    assert {
        row["user_id"]: row["ranked_items"] for row in recommendations.collect()
    } == {
        "user-1": ["c", "d"],
        "user-2": ["b", "c"],
    }


def test_category_top_k_orders_candidates_and_keeps_users_without_candidates(spark):
    from shoppilot.recommendations.popularity import recommend_from_category_candidates

    users = spark.createDataFrame(
        [
            ("tenant-1", "user-1"),
            ("tenant-1", "user-2"),
            ("tenant-1", "user-cold"),
            ("tenant-2", "user-1"),
        ],
        "tenant_id string, user_id string",
    )
    candidates = spark.createDataFrame(
        [
            ("tenant-1", "user-1", "product-c", 2, 1),
            ("tenant-1", "user-1", "product-b", 1, 2),
            ("tenant-1", "user-1", "product-a", 1, 1),
            ("tenant-1", "user-2", "product-d", 1, 1),
            ("tenant-2", "user-1", "product-x", 1, 1),
        ],
        """
        tenant_id string,
        user_id string,
        product_id string,
        category_preference_rank int,
        category_rank int
        """,
    )

    recommendations = recommend_from_category_candidates(users, candidates, k=2)

    assert {
        (row["tenant_id"], row["user_id"]): row["ranked_items"]
        for row in recommendations.collect()
    } == {
        ("tenant-1", "user-1"): ["product-a", "product-b"],
        ("tenant-1", "user-2"): ["product-d"],
        ("tenant-1", "user-cold"): [],
        ("tenant-2", "user-1"): ["product-x"],
    }


def test_category_recommendations_fill_from_global_without_duplicates_or_seen(spark):
    from shoppilot.recommendations.popularity import fill_category_recommendations_with_global

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    users = spark.createDataFrame(
        [
            ("tenant-1", "user-warm"),
            ("tenant-1", "user-seen"),
            ("tenant-1", "user-cold"),
            ("tenant-2", "user-cold"),
        ],
        "tenant_id string, user_id string",
    )
    category_recommendations = spark.createDataFrame(
        [
            ("tenant-1", "user-warm", ["product-b", "product-a"]),
            ("tenant-1", "user-seen", ["product-c"]),
            ("tenant-1", "user-cold", []),
        ],
        "tenant_id string, user_id string, ranked_items array<string>",
    )
    ranked_global = spark.createDataFrame(
        [
            ("tenant-1", "product-a", 1),
            ("tenant-1", "product-b", 2),
            ("tenant-1", "product-c", 3),
            ("tenant-1", "product-d", 4),
            ("tenant-1", "product-e", 5),
            ("tenant-2", "product-x", 1),
            ("tenant-2", "product-y", 2),
        ],
        "tenant_id string, product_id string, popularity_rank int",
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-seen", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-seen", "product-b"),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-cold", "product-a"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )

    recommendations = fill_category_recommendations_with_global(
        users, category_recommendations, ranked_global, seen_items, config, k=2
    )

    assert {
        (row["tenant_id"], row["user_id"]): row["ranked_items"]
        for row in recommendations.collect()
    } == {
        ("tenant-1", "user-warm"): ["product-b", "product-a"],
        ("tenant-1", "user-seen"): ["product-c", "product-d"],
        ("tenant-1", "user-cold"): ["product-a", "product-b"],
        ("tenant-2", "user-cold"): ["product-x", "product-y"],
    }


def test_global_popularity_recommendations_exclude_seen_then_fill_top_k(spark):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    users = spark.createDataFrame(
        [
            ("tenant-1", "user-seen"),
            ("tenant-1", "user-cold"),
            ("tenant-1", "user-all-seen"),
            ("tenant-2", "user-seen"),
        ],
        "tenant_id string, user_id string",
    )
    ranked_popularity = spark.createDataFrame(
        [
            ("tenant-1", "product-a", 1),
            ("tenant-1", "product-b", 2),
            ("tenant-1", "product-c", 3),
            ("tenant-1", "product-d", 4),
            ("tenant-2", "product-x", 1),
            ("tenant-2", "product-y", 2),
        ],
        "tenant_id string, product_id string, popularity_rank int",
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-seen", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-seen", "product-b"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-all-seen", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-all-seen", "product-b"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-all-seen", "product-c"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-all-seen", "product-d"),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-cold", "product-a"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )

    recommendations = recommend_global_popularity(
        users, ranked_popularity, seen_items, config, k=2
    )

    assert {
        (row["tenant_id"], row["user_id"]): row["ranked_items"]
        for row in recommendations.collect()
    } == {
        ("tenant-1", "user-seen"): ["product-c", "product-d"],
        ("tenant-1", "user-cold"): ["product-a", "product-b"],
        ("tenant-1", "user-all-seen"): [],
        ("tenant-2", "user-seen"): ["product-x", "product-y"],
    }
    assert recommendations.count() == 4


def test_global_popularity_fills_top_k_after_long_seen_prefix(spark):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    users = spark.createDataFrame(
        [("tenant-1", "user-seen"), ("tenant-1", "user-cold")],
        "tenant_id string, user_id string",
    )
    ranked_popularity = spark.createDataFrame(
        [("tenant-1", f"product-{rank}", rank) for rank in range(1, 9)],
        "tenant_id string, product_id string, popularity_rank int",
    )
    seen_items = spark.createDataFrame(
        [
            (
                "snapshot-v1",
                "signal-v1",
                "protocol-v1",
                "binary_unique_pair_v1",
                "tenant-1",
                "user-seen",
                f"product-{rank}",
            )
            for rank in range(1, 6)
        ] + [
            (
                "snapshot-v2",
                "signal-v1",
                "protocol-v1",
                "binary_unique_pair_v1",
                "tenant-1",
                "user-cold",
                "product-1",
            )
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )

    recommendations = recommend_global_popularity(
        users, ranked_popularity, seen_items, config, k=2
    )

    assert {
        row["user_id"]: row["ranked_items"] for row in recommendations.collect()
    } == {
        "user-seen": ["product-6", "product-7"],
        "user-cold": ["product-1", "product-2"],
    }


def test_global_popularity_recommendations_reject_nonpositive_k(spark):
    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    users = spark.createDataFrame(
        [("tenant-1", "user-1")],
        "tenant_id string, user_id string",
    )

    with pytest.raises(ValueError):
        recommend_global_popularity(users, users, users, config, k=0)
