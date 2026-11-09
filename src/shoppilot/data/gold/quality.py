"""Quality gates for the canonical Gold contract."""

from pyspark.sql import functions as F


VALID_ROLES = ("train", "validation", "test", "oot_stress")
VALID_USER_STATES = (
    "platform_cold",
    "als_cold_nonpositive_only",
    "short_positive_history",
    "warm_positive_2_4",
    "warm_positive_5_plus",
)
VALID_ITEM_STATES = ("known_to_als", "unknown_to_als")


def _sum_condition(condition, alias):
    return F.coalesce(
        F.sum(F.coalesce(condition, F.lit(False)).cast("long")),
        F.lit(0),
    ).alias(alias)


def calculate_fact_metrics(reviews_df, events_df) -> dict:
    review_count = reviews_df.count()
    row = events_df.agg(
        F.count("*").alias("event_count"),
        F.countDistinct(F.struct("tenant_id", "event_id")).alias(
            "distinct_event_key_count"
        ),
        _sum_condition(
            F.col("tenant_id").isNull()
            | F.col("event_id").isNull()
            | F.col("user_id").isNull()
            | F.col("product_id").isNull()
            | F.col("event_time").isNull()
            | F.col("rating").isNull(),
            "missing_required_event_value_count",
        ),
        _sum_condition(
            F.col("event_type") != "review_rating",
            "invalid_event_type_count",
        ),
        _sum_condition(
            F.col("rating").isNotNull()
            & ~F.col("rating").between(1.0, 5.0),
            "invalid_rating_count",
        ),
        _sum_condition(
            F.col("is_missing_current_catalog"),
            "orphan_event_count",
        ),
    ).first()
    event_count = row["event_count"]
    return {
        "silver_review_count": review_count,
        "event_count": event_count,
        "review_event_parity_difference_count": abs(
            review_count - event_count
        ),
        "duplicate_event_key_count": (
            event_count - row["distinct_event_key_count"]
        ),
        "missing_required_event_value_count": row[
            "missing_required_event_value_count"
        ],
        "invalid_event_type_count": row["invalid_event_type_count"],
        "invalid_rating_count": row["invalid_rating_count"],
        "orphan_event_count": row["orphan_event_count"],
    }


def calculate_signal_metrics(events_df, signals_df) -> dict:
    event_count = events_df.count()
    row = signals_df.agg(
        F.count("*").alias("signal_count"),
        F.countDistinct(
            F.struct("tenant_id", "event_id", "signal_version")
        ).alias("distinct_signal_key_count"),
        _sum_condition(
            (F.col("feedback_class") == "positive")
            & (
                ~F.col("is_training_eligible")
                | (F.col("preference") != 1.0)
                | (F.col("confidence") != 1.0)
            ),
            "invalid_positive_signal_count",
        ),
        _sum_condition(
            (F.col("feedback_class") != "positive")
            & (
                F.col("is_training_eligible")
                | F.col("preference").isNotNull()
                | F.col("confidence").isNotNull()
            ),
            "invalid_nonpositive_signal_count",
        ),
        _sum_condition(
            F.col("feedback_class") == "positive",
            "positive_signal_count",
        ),
        _sum_condition(
            F.col("feedback_class") == "neutral",
            "neutral_signal_count",
        ),
        _sum_condition(
            F.col("feedback_class") == "negative",
            "negative_signal_count",
        ),
    ).first()
    signal_count = row["signal_count"]
    return {
        "signal_count": signal_count,
        "event_signal_parity_difference_count": abs(
            event_count - signal_count
        ),
        "duplicate_signal_key_count": (
            signal_count - row["distinct_signal_key_count"]
        ),
        "invalid_positive_signal_count": row[
            "invalid_positive_signal_count"
        ],
        "invalid_nonpositive_signal_count": row[
            "invalid_nonpositive_signal_count"
        ],
        "positive_signal_count": row["positive_signal_count"],
        "neutral_signal_count": row["neutral_signal_count"],
        "negative_signal_count": row["negative_signal_count"],
    }


def calculate_assignment_metrics(signals_df, assignments_df) -> dict:
    signal_count = signals_df.count()
    row = assignments_df.agg(
        F.count("*").alias("assignment_count"),
        F.countDistinct(
            F.struct("tenant_id", "event_id", "protocol_version")
        ).alias("distinct_assignment_key_count"),
        _sum_condition(
            ~F.col("role").isin(*VALID_ROLES),
            "invalid_role_count",
        ),
        _sum_condition(
            F.col("is_evaluation_target")
            & (
                (F.col("role") == "train")
                | ~F.col("is_training_eligible")
            ),
            "invalid_target_role_count",
        ),
        _sum_condition(
            F.col("is_evaluation_target")
            & ~F.col("user_availability_state").isin(*VALID_USER_STATES),
            "invalid_target_user_state_count",
        ),
        _sum_condition(
            F.col("is_evaluation_target")
            & ~F.col("item_availability_state").isin(*VALID_ITEM_STATES),
            "invalid_target_item_state_count",
        ),
        _sum_condition(
            ~F.col("is_evaluation_target")
            & (
                F.col("history_depth_before_cutoff").isNotNull()
                | F.col("user_availability_state").isNotNull()
                | F.col("item_availability_state").isNotNull()
            ),
            "non_target_state_leak_count",
        ),
        _sum_condition(
            (F.col("role") == "train")
            & ~(F.col("event_time") < F.col("cutoff_time")),
            "train_boundary_violation_count",
        ),
        _sum_condition(
            (F.col("role") != "train")
            & (F.col("event_time") < F.col("cutoff_time")),
            "evaluation_boundary_violation_count",
        ),
        _sum_condition(
            F.col("role") == "train",
            "train_event_count",
        ),
        _sum_condition(
            F.col("role") == "validation",
            "validation_event_count",
        ),
        _sum_condition(F.col("role") == "test", "test_event_count"),
        _sum_condition(
            F.col("role") == "oot_stress",
            "oot_stress_event_count",
        ),
        _sum_condition(
            F.col("is_evaluation_target"),
            "evaluation_target_count",
        ),
    ).first()

    same_bucket_split_count = (
        assignments_df.groupBy(
            "tenant_id",
            "user_id",
            "event_time",
            "target_bucket_id",
        )
        .agg(F.countDistinct("role").alias("role_count"))
        .filter(F.col("role_count") > 1)
        .count()
    )
    assignment_count = row["assignment_count"]
    return {
        "assignment_count": assignment_count,
        "signal_assignment_parity_difference_count": abs(
            signal_count - assignment_count
        ),
        "duplicate_assignment_key_count": (
            assignment_count - row["distinct_assignment_key_count"]
        ),
        "invalid_role_count": row["invalid_role_count"],
        "invalid_target_role_count": row["invalid_target_role_count"],
        "invalid_target_user_state_count": row[
            "invalid_target_user_state_count"
        ],
        "invalid_target_item_state_count": row[
            "invalid_target_item_state_count"
        ],
        "non_target_state_leak_count": row["non_target_state_leak_count"],
        "train_boundary_violation_count": row[
            "train_boundary_violation_count"
        ],
        "evaluation_boundary_violation_count": row[
            "evaluation_boundary_violation_count"
        ],
        "same_timestamp_role_violation_count": same_bucket_split_count,
        "train_event_count": row["train_event_count"],
        "validation_event_count": row["validation_event_count"],
        "test_event_count": row["test_event_count"],
        "oot_stress_event_count": row["oot_stress_event_count"],
        "evaluation_target_count": row["evaluation_target_count"],
    }


def calculate_user_feature_metrics(user_features_df) -> dict:
    row = user_features_df.agg(
        F.count("*").alias("feature_row_count"),
        F.countDistinct(
            F.struct("tenant_id", "user_id", "feature_cutoff")
        ).alias("distinct_feature_key_count"),
        _sum_condition(
            F.col("last_event_time") >= F.col("feature_cutoff"),
            "future_event_feature_count",
        ),
        _sum_condition(
            F.col("last_positive_event_time").isNotNull()
            & (
                F.col("last_positive_event_time")
                >= F.col("feature_cutoff")
            ),
            "future_positive_feature_count",
        ),
        _sum_condition(
            F.col("all_interaction_count")
            != (
                F.col("positive_interaction_count")
                + F.col("neutral_interaction_count")
                + F.col("negative_interaction_count")
            ),
            "feature_class_count_mismatch_count",
        ),
        F.countDistinct("feature_cutoff").alias("feature_cutoff_count"),
    ).first()
    return {
        "feature_row_count": row["feature_row_count"],
        "duplicate_feature_key_count": (
            row["feature_row_count"] - row["distinct_feature_key_count"]
        ),
        "future_event_feature_count": row["future_event_feature_count"],
        "future_positive_feature_count": row[
            "future_positive_feature_count"
        ],
        "feature_class_count_mismatch_count": row[
            "feature_class_count_mismatch_count"
        ],
        "feature_cutoff_count": row["feature_cutoff_count"],
    }


def validate_gold_v2_metrics(metrics: dict) -> None:
    violation_names = (
        "review_event_parity_difference_count",
        "duplicate_event_key_count",
        "missing_required_event_value_count",
        "invalid_event_type_count",
        "invalid_rating_count",
        "event_signal_parity_difference_count",
        "duplicate_signal_key_count",
        "invalid_positive_signal_count",
        "invalid_nonpositive_signal_count",
        "signal_assignment_parity_difference_count",
        "duplicate_assignment_key_count",
        "invalid_role_count",
        "invalid_target_role_count",
        "invalid_target_user_state_count",
        "invalid_target_item_state_count",
        "non_target_state_leak_count",
        "train_boundary_violation_count",
        "evaluation_boundary_violation_count",
        "same_timestamp_role_violation_count",
        "duplicate_feature_key_count",
        "future_event_feature_count",
        "future_positive_feature_count",
        "feature_class_count_mismatch_count",
    )
    violations = [
        f"{name}: {metrics[name]}"
        for name in violation_names
        if metrics.get(name, 0) > 0
    ]
    if violations:
        raise ValueError("Gold v2 quality gate failed:\n" + "\n".join(violations))
