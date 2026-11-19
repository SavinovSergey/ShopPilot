import math

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.diagnostics import (
    calculate_catalog_coverage,
    calculate_novelty_at_k,
)


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderDiagnosticsTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def config():
    return ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )


def test_catalog_coverage_uses_distinct_top_k_known_items_per_tenant(spark, config):
    scored_rows = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["a", "b", "d"]),
            ("tenant-1", "user-2", ["outside", "c", "d"]),
            ("tenant-1", "user-3", []),
            ("tenant-2", "user-1", ["x", "x", "y"]),
            ("tenant-3", "user-1", ["orphan"]),
        ],
        "tenant_id string, user_id string, ranked_items array<string>",
    )
    artifact_scope = ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1")
    candidate_universe = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", product_id) for product_id in ("a", "b", "c", "d")]
        + [(*artifact_scope, "tenant-1", "a")]
        + [(*artifact_scope, "tenant-2", "x"), (*artifact_scope, "tenant-2", "y")]
        + [("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "outside")],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string
        """,
    )

    report = calculate_catalog_coverage(scored_rows, candidate_universe, config, k=2)
    rows = {row["tenant_id"]: row for row in report.collect()}

    assert len(rows) == 3
    assert rows["tenant-1"]["recommended_item_count"] == 3
    assert rows["tenant-1"]["candidate_item_count"] == 4
    assert rows["tenant-1"]["catalog_coverage"] == pytest.approx(0.75)
    assert rows["tenant-2"]["recommended_item_count"] == 1
    assert rows["tenant-2"]["candidate_item_count"] == 2
    assert rows["tenant-2"]["catalog_coverage"] == pytest.approx(0.5)
    assert rows["tenant-3"]["recommended_item_count"] == 0
    assert rows["tenant-3"]["candidate_item_count"] == 0
    assert rows["tenant-3"]["catalog_coverage"] is None


def test_catalog_coverage_rejects_nonpositive_k(spark, config):
    scored_rows = spark.createDataFrame(
        [("tenant-1", "user-1", ["a"])],
        "tenant_id string, user_id string, ranked_items array<string>",
    )
    candidate_universe = spark.createDataFrame(
        [("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "a")],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string
        """,
    )

    with pytest.raises(ValueError):
        calculate_catalog_coverage(scored_rows, candidate_universe, config, k=0)


def test_novelty_at_k_uses_train_popularity_and_user_macro_mean(spark, config):
    scored_rows = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["a", "b", "c"]),
            ("tenant-1", "user-2", ["c"]),
            ("tenant-1", "user-3", []),
            ("tenant-1", "user-4", ["outside"]),
            ("tenant-2", "user-1", ["a"]),
            ("tenant-3", "user-1", ["orphan"]),
        ],
        "tenant_id string, user_id string, ranked_items array<string>",
    )
    artifact_scope = ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1")
    item_popularity = spark.createDataFrame(
        [
            (*artifact_scope, "tenant-1", "a", 4),
            (*artifact_scope, "tenant-1", "b", 2),
            (*artifact_scope, "tenant-1", "c", 1),
            (*artifact_scope, "tenant-2", "a", 1),
            (*artifact_scope, "tenant-2", "b", 3),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "outside", 100),
        ],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string,
        positive_pair_count long
        """,
    )

    report = calculate_novelty_at_k(scored_rows, item_popularity, config, k=2)
    report_rows = report.collect()
    assert len(report_rows) == 3
    rows = {row["tenant_id"]: row for row in report_rows}

    first_user_novelty = (-math.log2(4 / 7) - math.log2(2 / 7)) / 2
    second_user_novelty = -math.log2(1 / 7)
    assert len(rows) == 3
    assert rows["tenant-1"]["recommended_user_count"] == 2
    assert rows["tenant-1"]["novelty"] == pytest.approx(
        (first_user_novelty + second_user_novelty) / 2
    )
    assert rows["tenant-2"]["recommended_user_count"] == 1
    assert rows["tenant-2"]["novelty"] == pytest.approx(2.0)
    assert rows["tenant-3"]["recommended_user_count"] == 0
    assert rows["tenant-3"]["novelty"] is None


def test_novelty_at_k_rejects_nonpositive_k(spark, config):
    scored_rows = spark.createDataFrame(
        [("tenant-1", "user-1", ["a"])],
        "tenant_id string, user_id string, ranked_items array<string>",
    )
    item_popularity = spark.createDataFrame(
        [("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "a", 1)],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string,
        positive_pair_count long
        """,
    )

    with pytest.raises(ValueError):
        calculate_novelty_at_k(scored_rows, item_popularity, config, k=0)
