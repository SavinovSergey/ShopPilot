"""Catalog-level diagnostics for recommendation baselines."""

from shoppilot.recommendations.config import ModelInputConfig
from pyspark.sql import functions as F


def calculate_catalog_coverage(
    scored_rows_df,
    candidate_universe_df,
    config: ModelInputConfig,
    k: int,
):
    """Measure how much of the train-known catalog reaches target users.

    Args:
        scored_rows_df: One row per validation target user with ordered
            ``ranked_items``. Include users even when they have no eligible
            targets.
        candidate_universe_df: Versioned train-known products by tenant.
        config: Snapshot, signal, protocol, and aggregation versions selecting
            the candidate universe.
        k: Positive prefix length of each recommendation list.

    Returns:
        A Spark DataFrame with one row per tenant in ``scored_rows_df`` and
        ``recommended_item_count``, ``candidate_item_count``, and
        ``catalog_coverage``. The numerator counts distinct train-known items
        shown in the first K positions to at least one target user. Coverage
        is null if the tenant has no candidates.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    filtered_candidate_universe = candidate_universe_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )
    top_k_items = scored_rows_df.withColumn(
        "top_k_items",
        F.slice(F.col("ranked_items"), 1, k)
    )
    exploded_top_k_items = top_k_items.select(
        "tenant_id",
        F.explode(F.col("top_k_items")).alias("product_id")
    )
    catalog_coverage_df = exploded_top_k_items.join(
        filtered_candidate_universe,
        on=["tenant_id", "product_id"],
        how="inner"
    )
    recommended_item_count = (
        catalog_coverage_df
        .select("tenant_id", "product_id")
        .distinct()
        .groupby("tenant_id")
        .count()
    ).withColumnRenamed("count", "recommended_item_count")
    candidate_item_count = (
        filtered_candidate_universe
        .select("tenant_id", "product_id")
        .distinct()
        .groupby("tenant_id")
        .count()
    ).withColumnRenamed("count", "candidate_item_count")
    catalog_coverage = scored_rows_df.select("tenant_id").distinct().join(
        recommended_item_count,
        on="tenant_id",
        how="left"
    ).join(
        candidate_item_count,
        on="tenant_id",
        how="left"
    )
    catalog_coverage = catalog_coverage.withColumn(
        "recommended_item_count",
        F.coalesce(F.col("recommended_item_count"), F.lit(0))
    ).withColumn(
        "candidate_item_count",
        F.coalesce(F.col("candidate_item_count"), F.lit(0))
    ).withColumn(
        "catalog_coverage",
        F.when(
            F.col("candidate_item_count") > 0, 
            F.col("recommended_item_count") / F.col("candidate_item_count")
        ).otherwise(F.lit(None))
    )
    return catalog_coverage


def calculate_novelty_at_k(
    scored_rows_df,
    item_popularity_df,
    config: ModelInputConfig,
    k: int,
):
    """Measure mean self-information of train-known recommendations at K.

    Args:
        scored_rows_df: One row per validation target user with ordered
            ``ranked_items``, including users without eligible targets.
        item_popularity_df: Versioned train-only ``positive_pair_count`` by
            tenant and product.
        config: Snapshot, signal, protocol, and aggregation versions selecting
            the popularity counts.
        k: Positive prefix length of each recommendation list.

    Returns:
        A Spark DataFrame with one row per tenant in ``scored_rows_df``,
        ``recommended_user_count``, and ``novelty``. Each train-known item has
        self-information ``-log2(item_count / tenant_total_count)``. Novelty
        is the macro mean of per-user means over users with at least one
        train-known recommendation in the first K positions. Unknown items
        do not contribute. If no user has a known recommendation, the count
        is zero and novelty is null.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    filtered_item_popularity = item_popularity_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )
    positive_pair_count_summary = filtered_item_popularity.groupby("tenant_id").agg(
        F.sum(F.col("positive_pair_count")).alias("positive_pair_count_summary")
    )
    top_k_items = scored_rows_df.withColumn(
        "top_k_items",
        F.slice(F.col("ranked_items"), 1, k)
    )
    exploded_top_k_items = top_k_items.select(
        "tenant_id",
        "user_id",
        F.explode(F.col("top_k_items")).alias("product_id")
    )
    joined_df = exploded_top_k_items.join(
        filtered_item_popularity,
        on=["tenant_id", "product_id"],
        how="inner"
    ).join(
        positive_pair_count_summary,
        on="tenant_id",
        how="inner"
    )
    novelty = joined_df.withColumn(
        "novelty",
        -F.log2(F.col("positive_pair_count") / F.col("positive_pair_count_summary"))
    ).groupby("tenant_id", "user_id").agg(
        F.avg(F.col("novelty")).alias("novelty")
    )
    novelty = novelty.groupby("tenant_id").agg(
        F.avg(F.col("novelty")).alias("novelty"),
        F.count("*").alias("recommended_user_count")
    )

    result = scored_rows_df.select("tenant_id").distinct().join(
        novelty,
        on="tenant_id",
        how="left"
    )
    result = result.withColumn(
        "recommended_user_count",
        F.coalesce(F.col("recommended_user_count"), F.lit(0))
    )
    return result
