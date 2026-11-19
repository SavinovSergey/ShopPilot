import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig, calculate_config_hash


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderEvaluationReportTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_evaluation_report_provenance_keeps_metrics_and_identifies_baseline(spark):
    from shoppilot.recommendations.evaluation_report import add_evaluation_provenance

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    report = spark.createDataFrame(
        [("tenant-1", 2, 0.5)],
        "tenant_id string, evaluable_user_count long, recall double",
    )

    global_row = add_evaluation_provenance(
        report, config, baseline="global_popularity", k=20
    ).first().asDict()
    category_row = add_evaluation_provenance(
        report, config, baseline="category_popularity", k=20
    ).first().asDict()

    for row in (global_row, category_row):
        assert row["tenant_id"] == "tenant-1"
        assert row["evaluable_user_count"] == 2
        assert row["recall"] == pytest.approx(0.5)
        assert row["snapshot_id"] == config.snapshot_id
        assert row["signal_version"] == config.signal_version
        assert row["protocol_version"] == config.protocol_version
        assert row["aggregation_version"] == config.aggregation_version
        assert row["cutoff"] == config.cutoff
        assert row["config_hash"] == calculate_config_hash(config)
        assert row["k"] == 20

    assert global_row["baseline"] == "global_popularity"
    assert global_row["catalog_temporality"] == "not_used"
    assert category_row["baseline"] == "category_popularity"
    assert category_row["catalog_temporality"] == "retrospective_current_catalog"
