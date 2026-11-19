from datetime import datetime, timezone
import math

import pytest
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.evaluation import (
    build_evaluation_rows,
    build_target_eligibility,
    build_validation_targets,
    score_evaluation_rows,
    summarize_evaluation_rows,
)
from shoppilot.recommendations.evaluation_pipeline import evaluate_global_popularity


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderEvaluationTests")
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


def make_assignments(spark, rows):
    return spark.createDataFrame(
        rows,
        """
        tenant_id string,
        event_id string,
        user_id string,
        product_id string,
        snapshot_id string,
        signal_version string,
        protocol_version string,
        role string,
        cutoff_time timestamp,
        is_training_eligible boolean,
        is_evaluation_target boolean,
        user_availability_state string
        """,
    )


def assignment(
    tenant_id,
    event_id,
    user_id,
    product_id,
    *,
    snapshot_id="snapshot-v1",
    signal_version="signal-v1",
    protocol_version="protocol-v1",
    role="validation",
    cutoff_time=datetime(2021, 1, 1, tzinfo=timezone.utc),
    is_training_eligible=True,
    is_evaluation_target=True,
    user_availability_state="platform_cold",
):
    return (
        tenant_id,
        event_id,
        user_id,
        product_id,
        snapshot_id,
        signal_version,
        protocol_version,
        role,
        cutoff_time,
        is_training_eligible,
        is_evaluation_target,
        user_availability_state,
    )


def test_validation_targets_are_distinct_sorted_and_tenant_scoped(spark, config):
    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "e1", "user-1", "product-b"),
            assignment("tenant-1", "e2", "user-1", "product-a"),
            assignment("tenant-1", "e3", "user-1", "product-b"),
            assignment("tenant-2", "e4", "user-1", "product-c"),
            assignment("tenant-1", "e5", "user-2", "product-d"),
        ],
    )

    targets = build_validation_targets(assignments, config)

    assert {
        (row["tenant_id"], row["user_id"]): row["relevant_items"]
        for row in targets.collect()
    } == {
        ("tenant-1", "user-1"): ["product-a", "product-b"],
        ("tenant-1", "user-2"): ["product-d"],
        ("tenant-2", "user-1"): ["product-c"],
    }
    assert targets.count() == 3


def test_validation_targets_preserve_user_availability_state(spark, config):
    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "e1", "user-1", "product-b", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e2", "user-1", "product-a", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e3", "user-2", "product-c", user_availability_state="platform_cold"),
            assignment("tenant-2", "e4", "user-1", "product-d", user_availability_state="short_positive_history"),
        ],
    )

    targets = build_validation_targets(assignments, config)

    assert {
        (row["tenant_id"], row["user_id"]): (
            row["user_availability_state"], row["relevant_items"]
        )
        for row in targets.collect()
    } == {
        ("tenant-1", "user-1"): ("warm_positive_2_4", ["product-a", "product-b"]),
        ("tenant-1", "user-2"): ("platform_cold", ["product-c"]),
        ("tenant-2", "user-1"): ("short_positive_history", ["product-d"]),
    }
    assert targets.count() == 3


def test_validation_targets_exclude_other_scopes_and_non_targets(spark, config):
    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "valid", "user-1", "product-a"),
            assignment("tenant-1", "train", "user-1", "product-b", role="train"),
            assignment("tenant-1", "test", "user-1", "product-c", role="test"),
            assignment("tenant-1", "neutral", "user-1", "product-d", is_training_eligible=False),
            assignment("tenant-1", "not-target", "user-1", "product-e", is_evaluation_target=False),
            assignment("tenant-1", "other-snapshot", "user-1", "product-f", snapshot_id="snapshot-v2"),
            assignment("tenant-1", "other-signal", "user-1", "product-g", signal_version="signal-v2"),
            assignment("tenant-1", "other-protocol", "user-1", "product-h", protocol_version="protocol-v2"),
            assignment("tenant-1", "other-cutoff", "user-1", "product-i", cutoff_time=datetime(2022, 1, 1, tzinfo=timezone.utc)),
        ],
    )

    targets = build_validation_targets(assignments, config)

    assert [(row["tenant_id"], row["user_id"], row["relevant_items"]) for row in targets.collect()] == [
        ("tenant-1", "user-1", ["product-a"]),
    ]


def test_target_eligibility_preserves_all_targets_and_separates_reasons(
    spark, config
):
    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "e1", "user-1", "product-a"),
            assignment("tenant-1", "e2", "user-1", "product-b"),
            assignment("tenant-1", "e3", "user-1", "product-c"),
            assignment("tenant-1", "e4", "user-2", "product-c"),
            assignment("tenant-2", "e5", "user-1", "product-a"),
        ],
    )
    targets = build_validation_targets(assignments, config)
    candidate_universe = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-a"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-b"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-d"),
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-2", "product-a"),
            ("snapshot-v2", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "product-c"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        product_id string
        """,
    )
    seen_items = spark.createDataFrame(
        [
            ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-a"),
            ("snapshot-v1", "signal-v2", "protocol-v1", "binary_unique_pair_v1", "tenant-1", "user-1", "product-b"),
        ],
        """
        snapshot_id string,
        signal_version string,
        protocol_version string,
        aggregation_version string,
        tenant_id string,
        user_id string,
        product_id string
        """,
    )

    eligibility = build_target_eligibility(
        targets, candidate_universe, seen_items, config
    )

    assert {
        (row["tenant_id"], row["user_id"]): {
            "relevant_items": row["relevant_items"],
            "eligible_items": row["eligible_items"],
            "seen_target_items": row["seen_target_items"],
            "unknown_target_items": row["unknown_target_items"],
        }
        for row in eligibility.collect()
    } == {
        ("tenant-1", "user-1"): {
            "relevant_items": ["product-a", "product-b", "product-c"],
            "eligible_items": ["product-b"],
            "seen_target_items": ["product-a"],
            "unknown_target_items": ["product-c"],
        },
        ("tenant-1", "user-2"): {
            "relevant_items": ["product-c"],
            "eligible_items": [],
            "seen_target_items": [],
            "unknown_target_items": ["product-c"],
        },
        ("tenant-2", "user-1"): {
            "relevant_items": ["product-a"],
            "eligible_items": ["product-a"],
            "seen_target_items": [],
            "unknown_target_items": [],
        },
    }


def test_evaluation_rows_preserve_denominator_and_tenant_scope(spark):
    eligibility = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["product-a", "product-b"], ["product-a"], ["product-b"], []),
            ("tenant-1", "user-2", ["product-c"], [], [], ["product-c"]),
            ("tenant-2", "user-1", ["product-x"], ["product-x"], [], []),
            ("tenant-2", "user-2", ["product-y"], ["product-y"], [], []),
            ("tenant-2", "user-3", ["product-z"], ["product-z"], [], []),
            ("tenant-3", "user-null", ["product-q"], None, [], ["product-q"]),
        ],
        """
        tenant_id string,
        user_id string,
        relevant_items array<string>,
        eligible_items array<string>,
        seen_target_items array<string>,
        unknown_target_items array<string>
        """,
    )
    recommendations = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["product-a"]),
            ("tenant-2", "user-1", ["product-y"]),
            ("tenant-2", "user-2", None),
            ("tenant-1", "not-a-target", ["product-c"]),
        ],
        "tenant_id string, user_id string, ranked_items array<string>",
    )

    rows = build_evaluation_rows(eligibility, recommendations)

    assert {
        (row["tenant_id"], row["user_id"]): (
            row["relevant_items"],
            row["eligible_items"],
            row["ranked_items"],
            row["is_evaluable"],
        )
        for row in rows.collect()
    } == {
        ("tenant-1", "user-1"): (["product-a", "product-b"], ["product-a"], ["product-a"], True),
        ("tenant-1", "user-2"): (["product-c"], [], [], False),
        ("tenant-2", "user-1"): (["product-x"], ["product-x"], ["product-y"], True),
        ("tenant-2", "user-2"): (["product-y"], ["product-y"], [], True),
        ("tenant-2", "user-3"): (["product-z"], ["product-z"], [], True),
        ("tenant-3", "user-null"): (["product-q"], None, [], False),
    }
    assert rows.count() == 6


def test_score_evaluation_rows_uses_eligible_targets_and_preserves_exclusions(spark):
    evaluation_rows = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["product-b", "product-a"], ["product-x", "product-b", "product-a"], True),
            ("tenant-1", "user-2", ["product-c"], [], True),
            ("tenant-2", "user-1", ["product-d"], ["product-d", "product-e"], True),
            ("tenant-2", "user-2", [], ["product-f"], False),
        ],
        """
        tenant_id string,
        user_id string,
        eligible_items array<string>,
        ranked_items array<string>,
        is_evaluable boolean
        """,
    )

    scored = score_evaluation_rows(evaluation_rows, k=2)
    rows = {
        (row["tenant_id"], row["user_id"]): row
        for row in scored.collect()
    }

    assert len(rows) == 4
    delayed_hit = rows[("tenant-1", "user-1")]
    assert delayed_hit["recall"] == pytest.approx(0.5)
    assert delayed_hit["hit_rate"] == pytest.approx(1.0)
    assert delayed_hit["ndcg"] == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )
    assert delayed_hit["mrr"] == pytest.approx(0.5)

    for metric in ("recall", "hit_rate", "ndcg", "mrr"):
        assert rows[("tenant-1", "user-2")][metric] == pytest.approx(0.0)
        assert rows[("tenant-2", "user-1")][metric] == pytest.approx(1.0)
        assert rows[("tenant-2", "user-2")][metric] is None

    truncated = score_evaluation_rows(evaluation_rows, k=1)
    first_user = truncated.filter(
        (truncated.tenant_id == "tenant-1") & (truncated.user_id == "user-1")
    ).first()
    assert first_user is not None
    for metric in ("recall", "hit_rate", "ndcg", "mrr"):
        assert first_user[metric] == pytest.approx(0.0)


def test_score_evaluation_rows_rejects_nonpositive_k(spark):
    evaluation_rows = spark.createDataFrame(
        [("tenant-1", "user-1", ["product-a"], ["product-a"], True)],
        "tenant_id string, user_id string, eligible_items array<string>, ranked_items array<string>, is_evaluable boolean",
    )

    with pytest.raises(ValueError):
        score_evaluation_rows(evaluation_rows, k=0)


def test_summarize_evaluation_rows_uses_user_macro_means_and_reports_exclusions(spark):
    scored_rows = spark.createDataFrame(
        [
            ("tenant-1", "user-1", ["a", "b", "c"], ["a", "b"], ["c"], [], True, 0.5, 1.0, 0.4, 0.5),
            ("tenant-1", "user-2", ["d", "e"], ["d"], [], ["e"], True, 0.0, 0.0, 0.0, 0.0),
            ("tenant-1", "user-3", ["f", "g"], [], ["f"], ["g"], False, None, None, None, None),
            ("tenant-2", "user-1", ["x"], [], [], ["x"], False, None, None, None, None),
        ],
        """
        tenant_id string,
        user_id string,
        relevant_items array<string>,
        eligible_items array<string>,
        seen_target_items array<string>,
        unknown_target_items array<string>,
        is_evaluable boolean,
        recall double,
        hit_rate double,
        ndcg double,
        mrr double
        """,
    )

    report = summarize_evaluation_rows(scored_rows)
    rows = {row["tenant_id"]: row for row in report.collect()}

    assert len(rows) == 2
    tenant_one = rows["tenant-1"]
    assert tenant_one["target_user_count"] == 3
    assert tenant_one["evaluable_user_count"] == 2
    assert tenant_one["excluded_user_count"] == 1
    assert tenant_one["total_target_count"] == 7
    assert tenant_one["eligible_target_count"] == 3
    assert tenant_one["seen_target_count"] == 2
    assert tenant_one["unknown_target_count"] == 2
    assert tenant_one["recall"] == pytest.approx(0.25)
    assert tenant_one["hit_rate"] == pytest.approx(0.5)
    assert tenant_one["ndcg"] == pytest.approx(0.2)
    assert tenant_one["mrr"] == pytest.approx(0.25)

    tenant_two = rows["tenant-2"]
    assert tenant_two["target_user_count"] == 1
    assert tenant_two["evaluable_user_count"] == 0
    assert tenant_two["excluded_user_count"] == 1
    assert tenant_two["total_target_count"] == 1
    assert tenant_two["eligible_target_count"] == 0
    assert tenant_two["seen_target_count"] == 0
    assert tenant_two["unknown_target_count"] == 1
    for metric in ("recall", "hit_rate", "ndcg", "mrr"):
        assert tenant_two[metric] is None


def test_summarize_evaluation_rows_separates_user_availability_slices(spark):
    scored_rows = spark.createDataFrame(
        [
            ("tenant-1", "user-1", "warm_positive_2_4", ["a", "b"], ["a"], ["b"], [], True, 1.0, 1.0, 1.0, 1.0),
            ("tenant-1", "user-2", "warm_positive_2_4", ["c"], ["c"], [], [], True, 0.0, 0.0, 0.0, 0.0),
            ("tenant-1", "user-3", "platform_cold", ["d"], [], [], ["d"], False, None, None, None, None),
            ("tenant-2", "user-1", "warm_positive_2_4", ["x", "y"], ["x", "y"], [], [], True, 0.5, 1.0, 0.5, 0.5),
        ],
        """
        tenant_id string,
        user_id string,
        user_availability_state string,
        relevant_items array<string>,
        eligible_items array<string>,
        seen_target_items array<string>,
        unknown_target_items array<string>,
        is_evaluable boolean,
        recall double,
        hit_rate double,
        ndcg double,
        mrr double
        """,
    )

    report = summarize_evaluation_rows(
        scored_rows, slice_columns=("user_availability_state",)
    )
    rows = {
        (row["tenant_id"], row["user_availability_state"]): row
        for row in report.collect()
    }

    assert len(rows) == 3
    warm = rows[("tenant-1", "warm_positive_2_4")]
    assert warm["target_user_count"] == 2
    assert warm["evaluable_user_count"] == 2
    assert warm["excluded_user_count"] == 0
    assert warm["total_target_count"] == 3
    assert warm["eligible_target_count"] == 2
    assert warm["seen_target_count"] == 1
    assert warm["unknown_target_count"] == 0
    assert warm["recall"] == pytest.approx(0.5)
    assert warm["hit_rate"] == pytest.approx(0.5)
    assert warm["ndcg"] == pytest.approx(0.5)
    assert warm["mrr"] == pytest.approx(0.5)

    cold = rows[("tenant-1", "platform_cold")]
    assert cold["target_user_count"] == 1
    assert cold["evaluable_user_count"] == 0
    assert cold["excluded_user_count"] == 1
    assert cold["unknown_target_count"] == 1
    for metric in ("recall", "hit_rate", "ndcg", "mrr"):
        assert cold[metric] is None

    other_tenant = rows[("tenant-2", "warm_positive_2_4")]
    assert other_tenant["target_user_count"] == 1
    assert other_tenant["recall"] == pytest.approx(0.5)
    assert other_tenant["hit_rate"] == pytest.approx(1.0)


def test_global_popularity_pipeline_preserves_targets_and_reports_slices(spark, config):
    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "e1", "user-warm", "a", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e2", "user-warm", "c", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e3", "user-cold", "a"),
            assignment("tenant-1", "e4", "user-excluded", "unknown"),
            assignment("tenant-2", "e5", "user-cold", "a"),
            assignment("tenant-1", "wrong-snapshot", "other", "c", snapshot_id="snapshot-v2"),
        ],
    )
    artifact_scope = ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1")
    candidate_universe = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", product_id) for product_id in ("a", "b", "c", "d")]
        + [(*artifact_scope, "tenant-2", "a")],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string
        """,
    )
    seen_items = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", "user-warm", "a")],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, user_id string,
        product_id string
        """,
    )
    item_popularity = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", "a", 4),
         (*artifact_scope, "tenant-1", "b", 3),
         (*artifact_scope, "tenant-1", "c", 2),
         (*artifact_scope, "tenant-1", "d", 1),
         (*artifact_scope, "tenant-2", "a", 1)],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string,
        positive_pair_count long
        """,
    )

    result = evaluate_global_popularity(
        assignments, candidate_universe, seen_items, item_popularity, config, k=2
    )
    scored = {
        (row["tenant_id"], row["user_id"]): row
        for row in result["scored_rows"].collect()
    }
    overall = {
        row["tenant_id"]: row for row in result["overall_report"].collect()
    }
    slices = {
        (row["tenant_id"], row["user_availability_state"]): row
        for row in result["user_availability_report"].collect()
    }

    assert len(scored) == 4
    assert scored[("tenant-1", "user-warm")]["baseline"] == "global_popularity"
    assert scored[("tenant-1", "user-warm")]["k"] == 2
    assert scored[("tenant-1", "user-warm")]["ranked_items"] == ["b", "c"]
    assert scored[("tenant-1", "user-warm")]["eligible_items"] == ["c"]
    assert scored[("tenant-1", "user-warm")]["seen_target_items"] == ["a"]
    assert scored[("tenant-1", "user-cold")]["ranked_items"] == ["a", "b"]
    assert scored[("tenant-1", "user-excluded")]["is_evaluable"] is False
    assert scored[("tenant-1", "user-excluded")]["recall"] is None
    assert scored[("tenant-2", "user-cold")]["ranked_items"] == ["a"]

    tenant_one = overall["tenant-1"]
    assert tenant_one["baseline"] == "global_popularity"
    assert tenant_one["catalog_temporality"] == "not_used"
    assert tenant_one["k"] == 2
    assert tenant_one["target_user_count"] == 3
    assert tenant_one["evaluable_user_count"] == 2
    assert tenant_one["excluded_user_count"] == 1
    assert tenant_one["total_target_count"] == 4
    assert tenant_one["eligible_target_count"] == 2
    assert tenant_one["seen_target_count"] == 1
    assert tenant_one["unknown_target_count"] == 1
    assert tenant_one["recall"] == pytest.approx(1.0)
    assert tenant_one["ndcg"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert tenant_one["mrr"] == pytest.approx(0.75)
    assert tenant_one["recommended_item_count"] == 3
    assert tenant_one["candidate_item_count"] == 4
    assert tenant_one["catalog_coverage"] == pytest.approx(0.75)
    warm_novelty = (-math.log2(3 / 10) - math.log2(2 / 10)) / 2
    cold_novelty = (-math.log2(4 / 10) - math.log2(3 / 10)) / 2
    assert tenant_one["recommended_user_count"] == 3
    assert tenant_one["novelty"] == pytest.approx(
        (warm_novelty + 2 * cold_novelty) / 3
    )
    assert overall["tenant-2"]["catalog_coverage"] == pytest.approx(1.0)
    assert overall["tenant-2"]["recommended_user_count"] == 1
    assert overall["tenant-2"]["novelty"] == pytest.approx(0.0)

    assert len(slices) == 3
    assert slices[("tenant-1", "warm_positive_2_4")]["baseline"] == "global_popularity"
    assert slices[("tenant-1", "warm_positive_2_4")]["target_user_count"] == 1
    cold = slices[("tenant-1", "platform_cold")]
    assert cold["target_user_count"] == 2
    assert cold["evaluable_user_count"] == 1
    assert cold["excluded_user_count"] == 1
    assert cold["recall"] == pytest.approx(1.0)
    assert slices[("tenant-2", "platform_cold")]["target_user_count"] == 1


def test_category_popularity_pipeline_prioritizes_category_and_reports_fallback(
    spark, config, monkeypatch
):
    from shoppilot.recommendations import evaluation_pipeline

    seen_users_by_builder = {}
    original_preferences = evaluation_pipeline.build_user_category_preferences
    original_candidates = evaluation_pipeline.build_user_category_candidates

    def build_preferences_for_targets(seen_items_df, products_df, model_config):
        seen_users_by_builder["preferences"] = {
            row["user_id"] for row in seen_items_df.select("user_id").distinct().collect()
        }
        return original_preferences(seen_items_df, products_df, model_config)

    def build_candidates_for_targets(
        preferences_df, ranked_categories_df, seen_items_df, model_config, k
    ):
        seen_users_by_builder["candidates"] = {
            row["user_id"] for row in seen_items_df.select("user_id").distinct().collect()
        }
        return original_candidates(
            preferences_df, ranked_categories_df, seen_items_df, model_config, k
        )

    monkeypatch.setattr(
        evaluation_pipeline,
        "build_user_category_preferences",
        build_preferences_for_targets,
    )
    monkeypatch.setattr(
        evaluation_pipeline,
        "build_user_category_candidates",
        build_candidates_for_targets,
    )

    assignments = make_assignments(
        spark,
        [
            assignment("tenant-1", "e1", "user-warm", "b", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e2", "user-warm", "d", user_availability_state="warm_positive_2_4"),
            assignment("tenant-1", "e3", "user-cold", "a"),
            assignment("tenant-1", "e4", "user-unknown", "unknown"),
            assignment("tenant-1", "wrong-snapshot", "other", "d", snapshot_id="snapshot-v2"),
        ],
    )
    artifact_scope = ("snapshot-v1", "signal-v1", "protocol-v1", "binary_unique_pair_v1")
    candidate_universe = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", product_id) for product_id in ("a", "b", "c", "d")],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string
        """,
    )
    seen_items = spark.createDataFrame(
        [
            (*artifact_scope, "tenant-1", "user-warm", "a"),
            (*artifact_scope, "tenant-1", "user-outside-validation", "b"),
        ],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, user_id string,
        product_id string
        """,
    )
    item_popularity = spark.createDataFrame(
        [(*artifact_scope, "tenant-1", "a", 4),
         (*artifact_scope, "tenant-1", "b", 3),
         (*artifact_scope, "tenant-1", "c", 2),
         (*artifact_scope, "tenant-1", "d", 1)],
        """
        snapshot_id string, signal_version string, protocol_version string,
        aggregation_version string, tenant_id string, product_id string,
        positive_pair_count long
        """,
    )
    products = spark.createDataFrame(
        [("tenant-1", "a", "hair"),
         ("tenant-1", "b", "skin"),
         ("tenant-1", "c", "skin"),
         ("tenant-1", "d", "hair")],
        "tenant_id string, product_id string, category string",
    )

    result = evaluation_pipeline.evaluate_category_popularity(
        assignments, candidate_universe, seen_items, item_popularity,
        products, config, k=2
    )

    assert seen_users_by_builder == {
        "preferences": {"user-warm"},
        "candidates": {"user-warm"},
    }

    scored = {
        row["user_id"]: row for row in result["scored_rows"].collect()
    }
    overall = result["overall_report"].first()
    slices = {
        row["user_availability_state"]: row
        for row in result["user_availability_report"].collect()
    }

    assert len(scored) == 3
    assert scored["user-warm"]["baseline"] == "category_popularity"
    assert scored["user-warm"]["k"] == 2
    assert scored["user-warm"]["ranked_items"] == ["d", "b"]
    assert scored["user-warm"]["eligible_items"] == ["b", "d"]
    assert scored["user-cold"]["ranked_items"] == ["a", "b"]
    assert scored["user-unknown"]["is_evaluable"] is False
    assert overall["target_user_count"] == 3
    assert overall["baseline"] == "category_popularity"
    assert overall["catalog_temporality"] == "retrospective_current_catalog"
    assert overall["k"] == 2
    assert overall["evaluable_user_count"] == 2
    assert overall["unknown_target_count"] == 1
    assert overall["recall"] == pytest.approx(1.0)
    assert overall["catalog_coverage"] == pytest.approx(0.75)
    assert overall["novelty"] is not None
    assert slices["warm_positive_2_4"]["target_user_count"] == 1
    assert slices["warm_positive_2_4"]["baseline"] == "category_popularity"
    assert slices["platform_cold"]["target_user_count"] == 2
