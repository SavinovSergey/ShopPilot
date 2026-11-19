import pyspark.sql.functions as F
from shoppilot.recommendations.config import calculate_config_hash


def add_evaluation_provenance(report_df, config, baseline, k):
    """Add provenance information to an evaluation report.

    Args:
        report_df: Evaluation report DataFrame.
        config: Evaluation configuration.
        baseline: Baseline name.
        k: Evaluation metric cutoff.

    Returns:
        Evaluation report DataFrame with provenance information.
    """
    report_df = report_df.withColumn(
        "snapshot_id",
        F.lit(config.snapshot_id)
    )
    report_df = report_df.withColumn(
        "signal_version",
        F.lit(config.signal_version)
    )
    report_df = report_df.withColumn(
        "protocol_version",
        F.lit(config.protocol_version)
    )
    report_df = report_df.withColumn(
        "aggregation_version",
        F.lit(config.aggregation_version)
    )
    report_df = report_df.withColumn(
        "cutoff",
        F.lit(config.cutoff)
    )
    report_df = report_df.withColumn(
        "config_hash",
        F.lit(calculate_config_hash(config))
    )
    report_df = report_df.withColumn(
        "baseline",
        F.lit(baseline)
    )
    report_df = report_df.withColumn(
        "k",
        F.lit(k)
    )

    temporality_by_baseline = {
        "global_popularity": "not_used",
        "category_popularity": "retrospective_current_catalog",
    }
    if baseline not in temporality_by_baseline:
        raise ValueError(f"Unknown baseline: {baseline}")

    report_df = report_df.withColumn(
        "catalog_temporality",
        F.lit(temporality_by_baseline[baseline]),
    )
    return report_df