"""Build leakage-safe, model-ready recommendation inputs."""

from pyspark.sql import functions as F
from pyspark.sql import Window


MAPPING_SCOPE_COLUMNS = [
    "snapshot_id",
    "signal_version",
    "protocol_version",
    "aggregation_version",
]

def select_training_events(
    signals_df,
    assignments_df,
    *,
    snapshot_id: str,
    signal_version: str,
    protocol_version: str,
    cutoff: str,
):
    """Select positive train events for one versioned evaluation configuration.

    The result must contain only events which belong to the requested snapshot,
    signal and protocol; are eligible for training; have the ``train`` role;
    and occurred strictly before ``cutoff``.
    """

    filtered_signals_df = signals_df.filter(
        (F.col("snapshot_id") == snapshot_id) &
        (F.col("signal_version") == signal_version) &
        (F.col("is_training_eligible")) &
        (F.col("event_time") < F.lit(cutoff).cast("timestamp"))
    )

    filtered_assignments_df = assignments_df.filter(
        (F.col("snapshot_id") == snapshot_id) &
        (F.col("signal_version") == signal_version) &
        (F.col("protocol_version") == protocol_version) &
        (F.col("role") == "train") &
        (F.col("is_training_eligible")) &
        (F.col("event_time") < F.lit(cutoff).cast("timestamp"))
    )

    join_keys = [
        "tenant_id",
        "event_id",
        "user_id",
        "product_id",
        "event_time",           # defense invariant
        "signal_version",       # defense invariant
        "snapshot_id",          # defense invariant
    ]

    assignments_keys = filtered_assignments_df.select(
        *join_keys,
        "protocol_version"
    )

    join_with_assignments = filtered_signals_df.join(
        assignments_keys,
        on=join_keys,
        how="inner"
    )

    return join_with_assignments


def aggregate_binary_pairs(
    training_events_df,
    aggregation_version: str = "binary_unique_pair_v1",
):
    """Aggregate positive train events to one uniformly weighted user-item pair."""
    if aggregation_version != "binary_unique_pair_v1":
        raise ValueError(f"Unsupported aggregation version: {aggregation_version}")

    grouped_training_events = training_events_df.groupBy(
        [
            "tenant_id",            # store
            "user_id",              # user-item matrix
            "product_id",           # user-item matrix
            "snapshot_id",          # data state
            "signal_version",       # behavior version
            "protocol_version",     # temporal split
        ]
    ).agg(
        F.count("*").alias("event_count") # count of positive train events
    )

    result_df = (
        grouped_training_events
        .withColumn("preference", F.lit(1.0))
        .withColumn("confidence", F.lit(1.0))
        .withColumn("aggregation_version", F.lit(aggregation_version))
    )
    return result_df


def build_user_mapping(pairs_df):
    """Build a user mapping from a pairs DataFrame."""
    user_keys = pairs_df.select(
        *MAPPING_SCOPE_COLUMNS,
        "tenant_id",
        "user_id",
    ).distinct()
    user_window = Window.partitionBy(*MAPPING_SCOPE_COLUMNS).orderBy("tenant_id", "user_id")
    user_mapping = user_keys.withColumn(
        "user_index",
        (F.row_number().over(user_window) - 1).cast("int")
    )
    return user_mapping


def build_item_mapping(pairs_df):
    """Build an item mapping from a pairs DataFrame."""
    item_keys = pairs_df.select(
        *MAPPING_SCOPE_COLUMNS,
        "tenant_id",
        "product_id",
    ).distinct()
    item_window = Window.partitionBy(*MAPPING_SCOPE_COLUMNS).orderBy("tenant_id", "product_id")
    item_mapping = item_keys.withColumn(
        "item_index",
        (F.row_number().over(item_window) - 1).cast("int")
    )
    return item_mapping


def attach_mapping_indices(pairs_df, user_mapping_df, item_mapping_df):
    """Attach collision-free integer user and item indices to train pairs."""
    user_join_keys = MAPPING_SCOPE_COLUMNS + ["tenant_id", "user_id"]
    pairs_users_df = pairs_df.join(
        user_mapping_df,
        on=user_join_keys,
        how="inner"
    )
    item_join_keys = MAPPING_SCOPE_COLUMNS + ["tenant_id", "product_id"]
    indexed_pairs = pairs_users_df.join(
        item_mapping_df,
        on=item_join_keys,
        how="inner"
    )
    return indexed_pairs


def calculate_mapping_coverage_metrics(
    pairs_df,
    user_mapping_df,
    item_mapping_df,
    indexed_pairs_df,
):
    """Measure whether mappings cover every train pair exactly once."""
    user_join_keys = MAPPING_SCOPE_COLUMNS + ["tenant_id", "user_id"]
    item_join_keys = MAPPING_SCOPE_COLUMNS + ["tenant_id", "product_id"]
    pair_count = pairs_df.count()
    indexed_pair_count = indexed_pairs_df.count()
    indexed_pair_parity_difference_count = abs(
        pair_count - indexed_pair_count
    )
    missing_user_mapping_count = (
        pairs_df.join(
            user_mapping_df.select(*user_join_keys),
            on=user_join_keys,
            how="left_anti"
        ).count()
    )
    missing_item_mapping_count = (
        pairs_df.join(
            item_mapping_df.select(*item_join_keys),
            on=item_join_keys,
            how="left_anti"
        ).count()
    )
    return {
        "pair_count": pair_count,
        "indexed_pair_count": indexed_pair_count,
        "indexed_pair_parity_difference_count": indexed_pair_parity_difference_count,
        "missing_user_mapping_count": missing_user_mapping_count,
        "missing_item_mapping_count": missing_item_mapping_count,
    }


def validate_mapping_coverage_metrics(metrics: dict) -> None:
    """Reject model inputs which lost or multiplied pairs during mapping joins."""

    violation_names = (
        "indexed_pair_parity_difference_count",
        "missing_user_mapping_count",
        "missing_item_mapping_count",
    )
    violations = [
        f"{name}: {metrics[name]}"
        for name in violation_names
        if metrics.get(name, 0) > 0
    ]
    if violations:
        raise ValueError(
            "Mapping coverage gate failed:\n"
            + "\n".join(violations)
        )


def build_seen_items(indexed_pairs_df):
    """Build train-only user-item history used by seen-item policies."""
    # Какие товары конкретный пользователь уже видел в train-сигнале
    seen_items_df = indexed_pairs_df.select(
        *MAPPING_SCOPE_COLUMNS,
        "tenant_id",
        "user_id",
        "product_id",
        "user_index",
        "item_index",
    )
    return seen_items_df


def build_candidate_universe(indexed_pairs_df):
    """Build the conservative train-known item candidate universe."""
    # Какие товары вообще известны модели на cutoff
    candidate_universe_df = indexed_pairs_df.select(
        *MAPPING_SCOPE_COLUMNS,
        "tenant_id",
        "product_id",
        "item_index",
    ).distinct()
    return candidate_universe_df


def build_item_popularity_counts(indexed_pairs_df):
    """Count positive train pairs per candidate item."""
    # Сколько уникальных положительных user-item pairs для каждого товара
    join_keys = MAPPING_SCOPE_COLUMNS + ["tenant_id", "product_id", "item_index"]
    user_item_popularity_counts_df = indexed_pairs_df.groupBy(
        *join_keys,
    ).agg(
        F.count("*").alias("positive_pair_count")
    )
    return user_item_popularity_counts_df
