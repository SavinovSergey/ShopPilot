"""Gold fact, signal, feature and evaluation transformations."""

from pyspark.sql import functions as F


DEFAULT_SIGNAL_NAME = "rating_implicit_uniform"
DEFAULT_SIGNAL_VERSION = "rating_implicit_uniform_v1"
DEFAULT_PROTOCOL = "global_calendar"
DEFAULT_PROTOCOL_VERSION = "global_calendar_v1"
DEFAULT_FEATURE_VERSION = "user_history_v1"


def _timestamp_literal(value: str):
    return F.lit(value).cast("timestamp")


def build_user_item_events(
    reviews_df,
    snapshot_id: str,
    source: str = "amazon_reviews_2023",
):
    """Create model-neutral event facts from every Silver review."""
    return reviews_df.select(
        "tenant_id",
        F.col("review_id").alias("event_id"),
        "user_id",
        "product_id",
        F.lit("review_rating").alias("event_type"),
        "event_time",
        F.col("rating").cast("double").alias("rating"),
        F.col("verified_purchase").cast("boolean").alias(
            "verified_purchase"
        ),
        F.col("helpful_vote").cast("long").alias("helpful_vote"),
        F.lit(source).alias("source"),
        "source_record_id",
        F.lit(snapshot_id).alias("snapshot_id"),
    )


def attach_current_catalog_status(events_df, products_df):
    """Add a current-catalog quality flag without dropping orphan events."""
    catalog_keys = products_df.select(
        "tenant_id",
        "product_id",
        F.lit(True).alias("_catalog_record_exists"),
        F.col("is_active").alias("current_catalog_is_active"),
    )
    return (
        events_df.join(catalog_keys, ["tenant_id", "product_id"], "left")
        .withColumn(
            "is_missing_current_catalog",
            F.col("_catalog_record_exists").isNull(),
        )
        .drop("_catalog_record_exists")
    )


def build_rating_signals(
    events_df,
    signal_version: str = DEFAULT_SIGNAL_VERSION,
):
    """Interpret review ratings for the first uniform implicit baseline."""
    feedback_class = (
        F.when(F.col("rating") >= 4, F.lit("positive"))
        .when(F.col("rating") == 3, F.lit("neutral"))
        .otherwise(F.lit("negative"))
    )
    is_positive = F.col("rating") >= 4

    return events_df.select(
        "tenant_id",
        "event_id",
        "user_id",
        "product_id",
        "event_time",
        F.lit(DEFAULT_SIGNAL_NAME).alias("signal_name"),
        F.lit(signal_version).alias("signal_version"),
        feedback_class.alias("feedback_class"),
        F.when(is_positive, F.lit(1.0))
        .otherwise(F.lit(None).cast("double"))
        .alias("preference"),
        F.when(is_positive, F.lit(1.0))
        .otherwise(F.lit(None).cast("double"))
        .alias("confidence"),
        is_positive.alias("is_training_eligible"),
        "snapshot_id",
    )


def build_user_features_at_cutoff(
    signals_df,
    feature_cutoff: str,
    feature_version: str = DEFAULT_FEATURE_VERSION,
):
    """Aggregate only the history that was available before one cutoff."""
    history = signals_df.filter(
        F.col("event_time") < _timestamp_literal(feature_cutoff)
    )
    positive_time = F.when(
        F.col("feedback_class") == "positive",
        F.col("event_time"),
    )

    return history.groupBy("tenant_id", "user_id").agg(
        F.count("*").alias("all_interaction_count"),
        F.countDistinct("event_time").alias(
            "all_distinct_event_time_count"
        ),
        F.sum((F.col("feedback_class") == "positive").cast("long")).alias(
            "positive_interaction_count"
        ),
        F.countDistinct(positive_time).alias(
            "positive_distinct_event_time_count"
        ),
        F.sum((F.col("feedback_class") == "neutral").cast("long")).alias(
            "neutral_interaction_count"
        ),
        F.sum((F.col("feedback_class") == "negative").cast("long")).alias(
            "negative_interaction_count"
        ),
        F.min("event_time").alias("first_event_time"),
        F.max("event_time").alias("last_event_time"),
        F.min(positive_time).alias("first_positive_event_time"),
        F.max(positive_time).alias("last_positive_event_time"),
        F.lit(feature_version).alias("feature_version"),
        _timestamp_literal(feature_cutoff).alias("feature_cutoff"),
        F.first("snapshot_id").alias("snapshot_id"),
    )


def _target_bucket_id():
    bucket_key = F.to_json(
        F.struct(
            F.col("tenant_id"),
            F.col("user_id"),
            F.unix_millis("event_time").alias("event_time_ms"),
        ),
        {"ignoreNullFields": "false"},
    )
    return F.sha2(bucket_key, 256)


def _user_availability_state():
    all_count = F.coalesce(F.col("all_history_event_count"), F.lit(0))
    positive_times = F.coalesce(
        F.col("positive_history_timestamp_count"),
        F.lit(0),
    )
    return (
        F.when(all_count == 0, F.lit("platform_cold"))
        .when(
            positive_times == 0,
            F.lit("als_cold_nonpositive_only"),
        )
        .when(positive_times == 1, F.lit("short_positive_history"))
        .when(positive_times <= 4, F.lit("warm_positive_2_4"))
        .otherwise(F.lit("warm_positive_5_plus"))
    )


def _evaluation_targets_for_window(
    signals_df,
    assigned_df,
    role: str,
    cutoff: str,
):
    cutoff_literal = _timestamp_literal(cutoff)
    history = signals_df.filter(F.col("event_time") < cutoff_literal)

    user_history = history.groupBy("tenant_id", "user_id").agg(
        F.count("*").alias("all_history_event_count"),
        F.countDistinct(
            F.when(
                F.col("feedback_class") == "positive",
                F.col("event_time"),
            )
        ).alias("positive_history_timestamp_count"),
    )
    known_items = (
        history.filter(F.col("is_training_eligible"))
        .select("tenant_id", "product_id")
        .distinct()
        .withColumn("_known_to_als", F.lit(True))
    )

    targets = assigned_df.filter(
        (F.col("role") == role) & F.col("is_training_eligible")
    )
    return (
        targets.join(user_history, ["tenant_id", "user_id"], "left")
        .join(known_items, ["tenant_id", "product_id"], "left")
        .withColumn("is_evaluation_target", F.lit(True))
        .withColumn(
            "history_depth_before_cutoff",
            F.coalesce(
                F.col("positive_history_timestamp_count"),
                F.lit(0),
            ),
        )
        .withColumn(
            "user_availability_state",
            _user_availability_state(),
        )
        .withColumn(
            "item_availability_state",
            F.when(
                F.coalesce(F.col("_known_to_als"), F.lit(False)),
                F.lit("known_to_als"),
            ).otherwise(F.lit("unknown_to_als")),
        )
        .drop(
            "all_history_event_count",
            "positive_history_timestamp_count",
            "_known_to_als",
        )
    )


def assign_global_calendar_roles(
    signals_df,
    train_end: str = "2021-01-01",
    validation_end: str = "2022-01-01",
    test_end: str = "2023-01-01",
    protocol_version: str = DEFAULT_PROTOCOL_VERSION,
):
    """Assign one global temporal role and point-in-time availability states."""
    train_end_literal = _timestamp_literal(train_end)
    validation_end_literal = _timestamp_literal(validation_end)
    test_end_literal = _timestamp_literal(test_end)

    role = (
        F.when(F.col("event_time") < train_end_literal, F.lit("train"))
        .when(
            F.col("event_time") < validation_end_literal,
            F.lit("validation"),
        )
        .when(F.col("event_time") < test_end_literal, F.lit("test"))
        .otherwise(F.lit("oot_stress"))
    )
    cutoff_time = (
        F.when(F.col("event_time") < train_end_literal, train_end_literal)
        .when(F.col("event_time") < validation_end_literal, train_end_literal)
        .when(F.col("event_time") < test_end_literal, validation_end_literal)
        .otherwise(test_end_literal)
    )

    assigned = signals_df.select(
        "tenant_id",
        "event_id",
        "user_id",
        "product_id",
        "event_time",
        "signal_version",
        F.lit(DEFAULT_PROTOCOL).alias("evaluation_protocol"),
        F.lit(protocol_version).alias("protocol_version"),
        role.alias("role"),
        cutoff_time.alias("cutoff_time"),
        _target_bucket_id().alias("target_bucket_id"),
        "is_training_eligible",
        "snapshot_id",
    )

    target_sets = [
        _evaluation_targets_for_window(
            signals_df,
            assigned,
            "validation",
            train_end,
        ),
        _evaluation_targets_for_window(
            signals_df,
            assigned,
            "test",
            validation_end,
        ),
        _evaluation_targets_for_window(
            signals_df,
            assigned,
            "oot_stress",
            test_end,
        ),
    ]
    targets = target_sets[0].unionByName(target_sets[1]).unionByName(
        target_sets[2]
    )

    non_targets = (
        assigned.join(
            targets.select("tenant_id", "event_id"),
            ["tenant_id", "event_id"],
            "left_anti",
        )
        .withColumn("is_evaluation_target", F.lit(False))
        .withColumn(
            "history_depth_before_cutoff",
            F.lit(None).cast("long"),
        )
        .withColumn(
            "user_availability_state",
            F.lit(None).cast("string"),
        )
        .withColumn(
            "item_availability_state",
            F.lit(None).cast("string"),
        )
    )
    return non_targets.unionByName(targets)
