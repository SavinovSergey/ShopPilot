"""Train-only popularity ranking for recommendation baselines."""

from shoppilot.recommendations.config import ModelInputConfig
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def rank_category_popularity(item_popularity_df, products_df, config):
    """Rank products within each category by train-only positive-pair count.

    Args:
        item_popularity_df: Versioned train-only counts by tenant and product.
        products_df: Current product catalog with ``tenant_id``,
            ``product_id``, and ``category``. Its categories are retrospective,
            not guaranteed to reflect product metadata at the train cutoff.
        config: Snapshot, signal, protocol, and aggregation versions to select.

    Returns:
        A Spark DataFrame with ``tenant_id``, ``category``, ``product_id``, 
        ``positive_pair_count``, and one-based ``category_rank``. 
        Higher counts rank first; product ID breaks ties in ascending order.
    """

    filtered_popularity_df = item_popularity_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    filtered_popularity_df = filtered_popularity_df.join(
        products_df,
        on = ["tenant_id", "product_id"],
        how = "inner"
    ).filter(F.col("category").isNotNull())

    window = (
        Window
        .partitionBy("tenant_id", "category")
        .orderBy(
            F.col("positive_pair_count").desc(), 
            F.col("product_id").asc()
        )
    )

    filtered_popularity_df = filtered_popularity_df.withColumn(
        "category_rank",
        F.row_number().over(window).cast("int")
    )

    return filtered_popularity_df.select(
        "tenant_id",
        "category",
        "product_id",
        "positive_pair_count",
        "category_rank",
    )


def build_user_category_preferences(seen_items_df, products_df, config):
    """Rank each user's categories by distinct train-seen product count.

    Args:
        seen_items_df: Versioned train-seen user-product pairs.
        products_df: Current product catalog with ``tenant_id``,
            ``product_id``, and ``category``. Its categories are retrospective,
            not guaranteed to reflect product metadata at the train cutoff.
        config: Snapshot, signal, protocol, and aggregation versions to select.

    Returns:
        A Spark DataFrame with one row per user and non-null category,
        ``seen_category_count`` and one-based ``category_preference_rank``.
        Higher counts rank first; category name breaks ties in ascending order.
    """

    filtered_seen_items_df = seen_items_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    filtered_seen_items_df = filtered_seen_items_df.join(
        products_df,
        on = ["tenant_id", "product_id"],
        how = "inner"
    ).filter(F.col("category").isNotNull())

    filtered_seen_items_df = filtered_seen_items_df.groupBy("tenant_id", "user_id", "category").agg(
        F.countDistinct(F.col("product_id")).alias("seen_category_count")
    )
    window = (
        Window
        .partitionBy("tenant_id", "user_id")
        .orderBy(
            F.col("seen_category_count").desc(), 
            F.col("category").asc()
        )
    )
    filtered_seen_items_df = filtered_seen_items_df.withColumn(
        "category_preference_rank",
        F.row_number().over(window).cast("int")
    )

    return filtered_seen_items_df.select(
        "tenant_id",
        "user_id",
        "category",
        "seen_category_count",
        "category_preference_rank",
    )


def build_user_category_candidates(preferences_df, ranked_categories_df, seen_items_df, config, k):
    """Build unseen product candidates from each user's preferred categories.

    Args:
        preferences_df: Output of ``build_user_category_preferences``.
        ranked_categories_df: Output of ``rank_category_popularity``.
        seen_items_df: Versioned train-seen user-product pairs.
        config: Snapshot, signal, protocol, and aggregation versions used to
            filter the seen-item artifact.
        k: Positive recommendation cutoff used to limit candidate ranks within
            each category before excluding products seen by users.

    Returns:
        A Spark DataFrame with ``tenant_id``, ``user_id``, ``category``,
        ``product_id``, ``category_preference_rank``, and ``category_rank``.
        Each row is an unseen product in a category preferred by the user.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    max_seen_by_category = preferences_df.groupBy("tenant_id", "category").agg(
        F.max("seen_category_count").alias("max_seen_by_category")
    )
    ranked_categories_df = ranked_categories_df.join(
        max_seen_by_category,
        on = ["tenant_id", "category"],
        how = "left"
    ).filter(F.col("category_rank") <= F.lit(k) + F.col("max_seen_by_category"))

    joined_df = preferences_df.join(
        ranked_categories_df,
        on = ["tenant_id", "category"],
        how = "inner"
    )
    filtered_seen_items_df = seen_items_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )
    joined_df = joined_df.join(
        filtered_seen_items_df,
        on = ["tenant_id", "user_id", "product_id"],
        how = "leftanti"
    )
    return joined_df.select(
        "tenant_id", 
        "user_id", 
        "category", 
        "product_id", 
        "category_preference_rank",
        "category_rank"
    )


def recommend_from_category_candidates(users_df, candidates_df, k):
    """Recommend the first K unseen products from each user's preferred categories.

    Args:
        users_df: One row per ``(tenant_id, user_id)`` to recommend for.
        candidates_df: Output of ``build_user_category_candidates``.
        k: Maximum number of products per user; must be positive.

    Returns:
        A Spark DataFrame with one row per input user and an ordered
        ``ranked_items`` array of up to K unseen products. Users without
        category candidates receive an empty array.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")
    
    filtered_candidates_df = candidates_df.join(
        users_df,
        on = ["tenant_id", "user_id"],
        how = "left_semi"
    )
    window = (
        Window
        .partitionBy("tenant_id", "user_id")
        .orderBy(
            F.col("category_preference_rank").asc(),
            F.col("category_rank").asc(),
            F.col("product_id").asc()
        )
    )
    filtered_candidates_df = filtered_candidates_df.withColumn(
        "rank",
        F.row_number().over(window).cast("int")
    )
    grouped_candidates_df = (
        filtered_candidates_df
        .filter(F.col("rank") <= k)
        .groupBy("tenant_id", "user_id")
        .agg(
            F.collect_list(
                F.struct(
                    F.col("category_preference_rank"), 
                    F.col("category_rank"), 
                    F.col("product_id")
                )
            ).alias("ranked_items")
        )
    )
    grouped_candidates_df = grouped_candidates_df.withColumn(
        "ranked_items",
        F.transform(
            F.sort_array(F.col("ranked_items"), asc = True), 
            lambda x: x["product_id"]
        )
    ).select(
        "tenant_id",
        "user_id",
        "ranked_items"
    )

    users_ranked_items_df = users_df.join(
        grouped_candidates_df,
        on = ["tenant_id", "user_id"],
        how = "left"
    ).withColumn(
        "ranked_items",
        F.coalesce(F.col("ranked_items"), F.lit([]))
    )
    return users_ranked_items_df


def rank_global_popularity(item_popularity_df, config: ModelInputConfig):
    """Rank products within each tenant by train-only positive-pair count.

    Args:
        item_popularity_df: Versioned train-only counts by tenant and product.
        config: Snapshot, signal, protocol, and aggregation versions to select.

    Returns:
        A Spark DataFrame with ``tenant_id``, ``product_id``,
        ``positive_pair_count``, and one-based ``popularity_rank``. Higher
        counts rank first; product ID breaks ties in ascending order.
    """

    filtered_popularity_df = item_popularity_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    window = (
        Window
        .partitionBy("tenant_id")
        .orderBy(
            F.col("positive_pair_count").desc(), 
            F.col("product_id").asc()
        )
    )

    ranked_popularity_df = filtered_popularity_df.withColumn(
        "popularity_rank",
        F.row_number().over(window).cast("int")
    )

    return ranked_popularity_df.select(
        "tenant_id",
        "product_id",
        "positive_pair_count",
        "popularity_rank",
    )


def recommend_global_popularity(
    users_df,
    ranked_popularity_df,
    seen_items_df,
    config: ModelInputConfig,
    k: int,
):
    """Recommend the first K train-popular unseen products to each user.

    Args:
        users_df: One row per ``(tenant_id, user_id)`` to recommend for.
        ranked_popularity_df: Output of ``rank_global_popularity``.
        seen_items_df: Versioned train-seen user-product pairs.
        config: Snapshot, signal, protocol, and aggregation versions used to
            select the matching seen-item artifact.
        k: Maximum number of products per user; must be positive.

    Returns:
        A Spark DataFrame with one row per input user and an ordered
        ``ranked_items`` array. Seen products are excluded before taking K.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    filtered_seen_items_df = seen_items_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    seen_products_by_user_df = (
        filtered_seen_items_df
        .join(users_df, on = ["tenant_id", "user_id"], how = "left_semi") # left_semi для оптимизации
        .groupBy("tenant_id", "user_id").agg(
            F.countDistinct(F.col("product_id")).alias("seen_products_count")
        )
    )
    max_seen_products_by_tenant = seen_products_by_user_df.groupBy("tenant_id").agg(
        F.max("seen_products_count").alias("max_seen_products_count")
    )
    filtered_ranked_popularity_df = ranked_popularity_df.join(
        max_seen_products_by_tenant,
        on = ["tenant_id"],
        how = "left"
    ).filter(F.col("popularity_rank") <= k + F.coalesce(F.col("max_seen_products_count"), F.lit(0)))

    user_product_pairs_df = users_df.join(
        filtered_ranked_popularity_df,
        on = ["tenant_id"],
        how = "inner"
    ).join(
        filtered_seen_items_df,
        on = ["tenant_id", "user_id", "product_id"],
        how = "leftanti"
    )

    window = (
        Window
        .partitionBy("tenant_id", "user_id")
        .orderBy(
            F.col("popularity_rank").asc()
        )
    )

    users_ranked_items_df = user_product_pairs_df.withColumn(
        "rank",
        F.row_number().over(window).cast("int")
    )
    
    top_k_items = (
        users_ranked_items_df
        .filter(F.col("rank") <= k)
        .groupBy("tenant_id", "user_id")
        .agg(
            F.collect_list(
                F.struct(F.col("popularity_rank"), F.col("product_id"))
            ).alias("ranked_items")
        )
    )

    top_k_items = top_k_items.withColumn(
        "ranked_items",
        F.transform(
            F.sort_array(F.col("ranked_items"), asc = True), 
            lambda x: x["product_id"]
        )
    )

    users_ranked_items_df = (
        users_df
        .join(
            top_k_items, 
            on = ["tenant_id", "user_id"], 
            how = "left"
        )
    )

    users_ranked_items_df = users_ranked_items_df.withColumn(
        "ranked_items",
        F.coalesce(F.col("ranked_items"), F.lit([]))
    )

    return users_ranked_items_df


def fill_category_recommendations_with_global(
    users_df, 
    category_recommendations_df, 
    ranked_global_df, 
    seen_items_df, 
    config, 
    k):
    """Fill category recommendations with global popularity candidates.

    Args:
        users_df: One row per ``(tenant_id, user_id)`` to recommend for.
        category_recommendations_df: Output of ``recommend_from_category_candidates``.
        ranked_global_df: Output of ``rank_global_popularity``.
        seen_items_df: Versioned train-seen user-product pairs.
        config: Snapshot, signal, protocol, and aggregation versions used to
            select the matching seen-item artifact.
        k: Maximum number of products per user; must be positive.

    Returns:
        A Spark DataFrame with one row per input user and an ordered
        ``ranked_items`` array of up to K unseen products. Global popularity
        fills missing category positions, including for cold users.

    Raises:
        ValueError: If ``k`` is not positive.
    """

    if k <= 0:
        raise ValueError("k must be positive")

    category_items_df = category_recommendations_df.withColumnRenamed(
        "ranked_items", 
        "category_items"
    )

    global_items_df = recommend_global_popularity(
        users_df,
        ranked_global_df,
        seen_items_df,
        config,
        k
    ).withColumnRenamed("ranked_items", "global_items")

    category_items_df = users_df.join(
        category_items_df,
        on = ["tenant_id", "user_id"],
        how = "left"
    )
    category_items_df = category_items_df.join(
        global_items_df,
        on = ["tenant_id", "user_id"],
        how = "left"
    )
    category_items_df = category_items_df.withColumn(
        "category_items", 
        F.coalesce(F.col("category_items"), F.lit([]))
    )
    category_items_df = category_items_df.withColumn(
        "global_items",
        F.filter(
            F.col("global_items"), 
            lambda product: ~F.array_contains(F.col("category_items"), product)
        )
    )
    category_items_df = category_items_df.withColumn(
        "ranked_items",
        F.slice(F.concat(F.col("category_items"), F.col("global_items")), 1, k)
    )
    return category_items_df.select(
        "tenant_id",
        "user_id",
        "ranked_items"
    )
    
