"""Quality metrics and leakage gates for recommender model inputs."""

from shoppilot.recommendations.config import ModelInputConfig
from pyspark.sql import functions as F


EVENT_ID_COLUMNS = [
    "tenant_id",
    "event_id",
    "user_id",
    "product_id",
    "event_time",
    "snapshot_id",
    "signal_version",
    "protocol_version"
]

def calculate_training_leakage_metrics(
    training_events_df,
    assignments_df,
    config: ModelInputConfig,
) -> dict:
    """Measure whether selected training events violate temporal boundaries."""
    training_event_count = training_events_df.count()
    assignments_unique = assignments_df.dropDuplicates(subset=[*EVENT_ID_COLUMNS, "role"])
    non_train_events = assignments_unique.filter(F.col("role") != config.training_role)
    non_train_events = non_train_events.select(*EVENT_ID_COLUMNS).distinct()
    intersection = training_events_df.join(
        non_train_events,
        on=EVENT_ID_COLUMNS,
        how="inner"
    ).count()

    events_after_cutoff = training_events_df.filter(
        F.col("event_time") >= F.lit(config.cutoff).cast("timestamp")
    ).count()

    diff_event_count = training_events_df.filter(
        (F.col("snapshot_id") != config.snapshot_id)
        | (F.col("signal_version") != config.signal_version)
        | (F.col("protocol_version") != config.protocol_version)
    ).count()

    missing_training_assignments_count = training_events_df.join(
        assignments_unique
        .filter(F.col("role") == config.training_role)
        .select(*EVENT_ID_COLUMNS)
        .distinct(),
        on=EVENT_ID_COLUMNS,
        how="left_anti"
    ).count()

    return {
        "training_event_count": training_event_count,
        "future_role_overlap_count": intersection,
        "at_or_after_cutoff_count": events_after_cutoff,
        "wrong_scope_count": diff_event_count,
        "missing_training_assignment_count": missing_training_assignments_count
    }


def validate_training_leakage_metrics(metrics: dict) -> None:
    """Reject training events which violate version or temporal boundaries."""
    for metric_name, metric_value in metrics.items():
        if metric_name == "training_event_count":
            continue
        if metric_value > 0:
            raise ValueError(f"{metric_name}: {metric_value}")

