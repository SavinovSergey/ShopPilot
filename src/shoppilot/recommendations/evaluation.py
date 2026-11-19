"""Prepare user-level targets for recommender evaluation."""

from shoppilot.recommendations.config import ModelInputConfig
from pyspark.sql import functions as F
from typing import Tuple


def build_validation_targets(assignments_df, config: ModelInputConfig):
    """Group positive validation targets for one model-input configuration.

    Args:
        assignments_df: Gold evaluation assignments containing target flags,
            temporal role, cutoff, version fields, tenant, user, and product IDs.
        config: Snapshot, signal, protocol, and cutoff to evaluate.

    Returns:
        A Spark DataFrame with one row per ``(tenant_id, user_id, user_availability_state)`` 
        and a sorted array of distinct product IDs in ``relevant_items``.
    """
    validation_targets = assignments_df.filter(
        (F.col("role") == "validation")
        & (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("cutoff_time") == F.lit(config.cutoff))
        & (F.col("is_training_eligible"))
        & (F.col("is_evaluation_target"))
    )
    grouped_validation_targets = validation_targets.groupBy(
        "tenant_id", "user_id", "user_availability_state"
    ).agg(
        F.sort_array(
            F.collect_set(F.col("product_id")), 
            asc=True
        ).alias("relevant_items")
    )
    return grouped_validation_targets


def build_target_eligibility(
    targets_df,
    candidate_universe_df,
    seen_items_df,
    config: ModelInputConfig,
):
    """Classify validation targets under the train-only candidate policy.

    Args:
        targets_df: User-level targets from ``build_validation_targets``.
        candidate_universe_df: Train-known products, scoped by tenant and
            model-input versions.
        seen_items_df: Train-seen user-product pairs, scoped by tenant and
            model-input versions.
        config: Snapshot, signal, protocol, and aggregation versions used to
            select the matching train artifacts.

    Returns:
        A Spark DataFrame with one row per ``(tenant_id, user_id)``. It keeps
        ``relevant_items`` and adds sorted arrays ``eligible_items``,
        ``seen_target_items``, and ``unknown_target_items``. Each target
        belongs to exactly one of the three classification arrays.
    """

    exploded_targets = targets_df.select(
        "tenant_id",
        "user_id",
        F.explode(F.col("relevant_items")).alias("product_id")
    )

    filtered_candidate_universe = candidate_universe_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    filtered_seen_items = seen_items_df.filter(
        (F.col("snapshot_id") == F.lit(config.snapshot_id)) 
        & (F.col("signal_version") == F.lit(config.signal_version))
        & (F.col("protocol_version") == F.lit(config.protocol_version))
        & (F.col("aggregation_version") == F.lit(config.aggregation_version))
    )

    candidates = (
        filtered_candidate_universe
        .select("tenant_id", "product_id")
        .distinct()
        .withColumn("_is_candidate", F.lit(True))
    )

    seen_items = (
        filtered_seen_items
        .select("tenant_id", "user_id", "product_id")
        .distinct()
        .withColumn("_is_seen", F.lit(True))
    )

    target_eligibility = exploded_targets.join(
        candidates,
        on=["tenant_id", "product_id"],
        how="left"
    ).join(
        seen_items,
        on=["tenant_id", "user_id", "product_id"],
        how="left"
    )

    target_eligibility = target_eligibility.withColumn(
        "eligible_status",
        F.when(F.col("_is_seen"), "seen").otherwise(
            F.when(F.col("_is_candidate"), "eligible").otherwise(
                "unknown"
            )
        )
    )

    grouped_eligibility = target_eligibility.groupBy(
        "tenant_id", "user_id"
    ).agg(
        F.sort_array(
            F.collect_set(
                F.when(
                    F.col("eligible_status") == "seen", 
                    F.col("product_id")
                )
            ), 
            asc=True
        ).alias("seen_target_items"),
        F.sort_array(
            F.collect_set(
                F.when(
                    F.col("eligible_status") == "unknown", 
                    F.col("product_id")
                )
            ), 
            asc=True
        ).alias("unknown_target_items"),
        F.sort_array(
            F.collect_set(
                F.when(
                    F.col("eligible_status") == "eligible", 
                    F.col("product_id")
                )
            ), 
            asc=True
        ).alias("eligible_items"),
    )

    results = targets_df.join(
        grouped_eligibility,
        on=["tenant_id", "user_id"],
        how="left"
    )
    return results


def build_evaluation_rows(eligibility_df, recommendations_df):
    """Attach recommendations without dropping validation target users.

    Args:
        eligibility_df: One row per tenant and target user, with relevant,
            eligible, seen-target, and unknown-target item arrays.
        recommendations_df: One row per tenant and user, with an ordered
            ``ranked_items`` array. Some target users may be absent.

    Returns:
        A Spark DataFrame retaining every eligibility row and its target
        arrays, with ``ranked_items`` set to an empty array when missing or
        null and ``is_evaluable`` true only for nonempty ``eligible_items``.
    """

    evaluation_rows = eligibility_df.join(
        recommendations_df,
        on = ["tenant_id", "user_id"],
        how = "left"
    )
    evaluation_rows = evaluation_rows.withColumn(
        "ranked_items",
        F.coalesce(F.col("ranked_items"), F.lit([]))
    )
    evaluation_rows = evaluation_rows.withColumn(
        "is_evaluable",
        ((F.col("eligible_items").isNotNull()) 
        & (F.size(F.col("eligible_items")) > 0)).cast("boolean")
    )
    return evaluation_rows


def score_evaluation_rows(evaluation_rows_df, k: int):
    """Score each validation user against reachable targets at K.

    Args:
        evaluation_rows_df: Output of ``build_evaluation_rows`` with
            ``eligible_items``, ordered ``ranked_items``, and ``is_evaluable``.
        k: Number of leading recommendations to evaluate; must be positive.

    Returns:
        A Spark DataFrame retaining every input row and adding ``recall``,
        ``hit_rate``, ``ndcg``, and ``mrr`` as double columns. Evaluable users
        without recommendations receive zeros; non-evaluable users receive
        null metrics so they do not enter quality-metric denominators.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError("k must be positive")

    # evaluation_rows_df = evaluation_rows_df.filter(F.col("is_evaluable"))
    evaluation_top_k_items = evaluation_rows_df.withColumn(
        "top_k_items",
        F.slice(F.col("ranked_items"), 1, k)
    )
    # count the number of eligible items in the top_k_items
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "eligible_hit_count",
        F.size(F.array_intersect(F.col("top_k_items"), F.col("eligible_items")))
    )
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "recall",
        F.when(
            F.col("is_evaluable"), 
            F.col("eligible_hit_count") / F.size(F.col("eligible_items"))
        )
    )
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "hit_rate",
        F.when(
            F.col("is_evaluable"), 
            (F.col("eligible_hit_count") > 0).cast("double")
        )
    )

    hit_positions = F.transform(
        F.col("top_k_items"), 
        lambda item, index: F.when(
            F.array_contains(F.col("eligible_items"), item),
            index + 1
        ),
    )
    first_hit_position = F.array_min(hit_positions)
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "mrr",
        F.when(
            F.col("is_evaluable"), 
            F.coalesce(1.0 / first_hit_position, F.lit(0.0))
        )
    )
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "dcg",
        F.when(
            F.col("is_evaluable"),
            F.aggregate(
                F.transform(
                    F.col("top_k_items"),
                    lambda item, index: F.when(
                        F.array_contains(F.col("eligible_items"), item),
                        1 / F.log2(index + 2)
                    ).otherwise(F.lit(0.0))
                ),
                F.lit(0.0), 
                lambda acc, x: acc + x,
            )
        )
    )
    ideal_positions = F.filter(
        F.sequence(F.lit(1), F.lit(k)),
        lambda position: position <= F.size(F.col("eligible_items")),
    )
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "idcg",
        F.when(
            F.col("is_evaluable"),
            F.aggregate(
                ideal_positions,
                F.lit(0.0),
                lambda acc, index: acc + 1 / F.log2(index + 1)
            )
        )
    )
    evaluation_top_k_items = evaluation_top_k_items.withColumn(
        "ndcg",
        F.when(
            F.col("is_evaluable"),
            F.col("dcg") / F.col("idcg")
        )
    ).drop("dcg", "idcg")
    
    return evaluation_top_k_items


def summarize_evaluation_rows(scored_rows_df, slice_columns: Tuple[str, ...] = ()):
    """Summarize recommendation quality and target eligibility per tenant.

    Args:
        scored_rows_df: One scored row per tenant and validation user, with
            target-classification arrays, ``is_evaluable``, and four ranking
            metrics from ``score_evaluation_rows``.
        slice_columns: Columns to slice the data by.

    Returns:
        A Spark DataFrame with one row per tenant. It contains target,
        evaluable, and excluded user counts; total, eligible, seen, and
        unknown target-item counts; and user-macro means of ``recall``,
        ``hit_rate``, ``ndcg``, and ``mrr``. Mean metrics are null when a tenant
        has no evaluable users.
        If slice_columns is not empty, the result is grouped by the 
        slice_columns and tenant_id. 
        Otherwise, the result is grouped by tenant_id.
    """

    grouped_scored_rows = scored_rows_df.groupBy("tenant_id", *slice_columns).agg(
        F.count("*").alias("target_user_count"),
        F.sum(
            F.when(F.col("is_evaluable"), F.lit(1)).otherwise(F.lit(0))
        ).alias("evaluable_user_count"),
        F.sum(
            F.coalesce(F.size(F.col("relevant_items")), F.lit(0))
        ).alias("total_target_count"),
        F.sum(
            F.coalesce(F.size(F.col("eligible_items")), F.lit(0))
        ).alias("eligible_target_count"),
        F.sum(
            F.coalesce(F.size(F.col("seen_target_items")), F.lit(0))
        ).alias("seen_target_count"),
        F.sum(
            F.coalesce(F.size(F.col("unknown_target_items")), F.lit(0))
        ).alias("unknown_target_count"),
        F.avg("recall").alias("recall"),
        F.avg("hit_rate").alias("hit_rate"),
        F.avg("ndcg").alias("ndcg"),
        F.avg("mrr").alias("mrr"),
    )

    grouped_scored_rows = grouped_scored_rows.withColumn(
        "excluded_user_count",
        F.col("target_user_count") - F.col("evaluable_user_count")
    )
    
    return grouped_scored_rows
