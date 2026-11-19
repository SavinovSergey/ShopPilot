import pytest
from unittest.mock import Mock
from pyspark.sql import SparkSession

from shoppilot.recommendations.config import ModelInputConfig, calculate_config_hash


@pytest.fixture(scope="module")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("RecommenderEvaluationJobTests")
        .config("spark.sql.caseSensitive", "true")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.log.level", "WARN")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    yield session
    session.stop()


def test_load_evaluation_inputs_uses_versioned_gold_and_model_artifacts(spark, tmp_path):
    from shoppilot.jobs.recommender_evaluation import load_evaluation_inputs

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    gold_root = tmp_path / "data/lake/gold/snapshots" / config.snapshot_id
    model_root = (
        tmp_path / "data/lake/model_inputs" / config.snapshot_id
        / calculate_config_hash(config) / "artifacts"
    )
    spark.createDataFrame(
        [("tenant-1", "user-1", "product-a")],
        "tenant_id string, user_id string, product_id string",
    ).write.parquet(str(gold_root / "evaluation_assignments" / config.protocol_version))
    spark.createDataFrame(
        [("tenant-1", "product-a", "skin")],
        "tenant_id string, product_id string, category string",
    ).write.parquet(str(gold_root / "product_documents"))
    spark.createDataFrame(
        [("tenant-1", "product-a")],
        "tenant_id string, product_id string",
    ).write.parquet(str(model_root / "candidate_universe"))
    spark.createDataFrame(
        [("tenant-1", "user-1", "product-a")],
        "tenant_id string, user_id string, product_id string",
    ).write.parquet(str(model_root / "seen_items"))
    spark.createDataFrame(
        [("tenant-1", "product-a", 3)],
        "tenant_id string, product_id string, positive_pair_count long",
    ).write.parquet(str(model_root / "item_popularity"))

    inputs = load_evaluation_inputs(spark, config, project_root=tmp_path)

    assert set(inputs) == {
        "assignments",
        "products",
        "candidate_universe",
        "seen_items",
        "item_popularity",
    }
    assert inputs["assignments"].first()["user_id"] == "user-1"
    assert inputs["products"].first()["category"] == "skin"
    assert inputs["candidate_universe"].first()["product_id"] == "product-a"
    assert inputs["seen_items"].first()["user_id"] == "user-1"
    assert inputs["item_popularity"].first()["positive_pair_count"] == 3


def test_run_recommender_evaluation_job_calls_both_baselines(monkeypatch, tmp_path):
    from shoppilot.jobs import recommender_evaluation

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    spark = object()
    inputs = {
        "assignments": object(),
        "products": object(),
        "candidate_universe": object(),
        "seen_items": object(),
        "item_popularity": object(),
    }
    global_results = {"overall_report": object()}
    category_results = {"overall_report": object()}
    load_inputs = Mock(return_value=inputs)
    evaluate_global = Mock(return_value=global_results)
    evaluate_category = Mock(return_value=category_results)
    output_paths = object()
    build_paths = Mock(return_value=output_paths)
    save_reports = Mock()
    monkeypatch.setattr(recommender_evaluation, "load_evaluation_inputs", load_inputs)
    monkeypatch.setattr(recommender_evaluation, "evaluate_global_popularity", evaluate_global)
    monkeypatch.setattr(recommender_evaluation, "evaluate_category_popularity", evaluate_category)
    monkeypatch.setattr(recommender_evaluation, "build_evaluation_output_paths", build_paths)
    monkeypatch.setattr(recommender_evaluation, "save_evaluation_reports", save_reports)

    results = recommender_evaluation.run_recommender_evaluation_job(
        spark, config, k=20, project_root=tmp_path
    )

    load_inputs.assert_called_once_with(spark, config, tmp_path)
    evaluate_global.assert_called_once_with(
        inputs["assignments"],
        inputs["candidate_universe"],
        inputs["seen_items"],
        inputs["item_popularity"],
        config,
        20,
    )
    evaluate_category.assert_called_once_with(
        inputs["assignments"],
        inputs["candidate_universe"],
        inputs["seen_items"],
        inputs["item_popularity"],
        inputs["products"],
        config,
        20,
    )
    assert results == {
        "global_popularity": global_results,
        "category_popularity": category_results,
    }
    build_paths.assert_called_once_with(config, 20, tmp_path)
    save_reports.assert_called_once_with(results, output_paths)


def test_build_evaluation_output_paths_is_versioned_and_has_report_paths(tmp_path):
    from shoppilot.jobs.recommender_evaluation import build_evaluation_output_paths

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    output_root = (
        tmp_path
        / "data/lake/recommender_evaluations"
        / config.snapshot_id
        / calculate_config_hash(config)
        / "k=20"
    )

    paths = build_evaluation_output_paths(config, k=20, project_root=tmp_path)

    assert paths == {
        baseline: {
            report_name: output_root / baseline / report_name
            for report_name in (
                "scored_rows",
                "overall_report",
                "user_availability_report",
            )
        }
        for baseline in ("global_popularity", "category_popularity")
    }
    assert not output_root.exists()


def test_save_evaluation_reports_writes_each_report_without_overwrite(tmp_path):
    from shoppilot.jobs.recommender_evaluation import (
        build_evaluation_output_paths,
        save_evaluation_reports,
    )

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    paths = build_evaluation_output_paths(config, k=20, project_root=tmp_path)
    results = {
        baseline: {report_name: Mock() for report_name in reversed(tuple(report_paths))}
        for baseline, report_paths in paths.items()
    }

    save_evaluation_reports(results, paths)

    for baseline, report_paths in paths.items():
        for report_name, path in report_paths.items():
            report_df = results[baseline][report_name]
            report_df.write.mode.assert_called_once_with("errorifexists")
            report_df.write.mode.return_value.parquet.assert_called_once_with(str(path))


def test_save_evaluation_reports_rejects_existing_path_before_any_write(tmp_path):
    from shoppilot.jobs.recommender_evaluation import (
        build_evaluation_output_paths,
        save_evaluation_reports,
    )

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    paths = build_evaluation_output_paths(config, k=20, project_root=tmp_path)
    results = {
        baseline: {report_name: Mock() for report_name in report_paths}
        for baseline, report_paths in paths.items()
    }
    paths["category_popularity"]["user_availability_report"].mkdir(parents=True)

    with pytest.raises(FileExistsError):
        save_evaluation_reports(results, paths)

    for reports in results.values():
        for report_df in reports.values():
            report_df.write.mode.assert_not_called()


def test_save_evaluation_reports_publishes_parquet_and_preserves_existing_data(
    spark, tmp_path
):
    from shoppilot.jobs.recommender_evaluation import (
        build_evaluation_output_paths,
        save_evaluation_reports,
    )

    config = ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )
    paths = build_evaluation_output_paths(config, k=20, project_root=tmp_path)
    results = {
        baseline: {
            report_name: spark.createDataFrame(
                [(baseline, report_name)], "baseline string, report_name string"
            )
            for report_name in report_paths
        }
        for baseline, report_paths in paths.items()
    }

    save_evaluation_reports(results, paths)

    for baseline, report_paths in paths.items():
        for report_name, path in report_paths.items():
            assert (path / "_SUCCESS").exists()
            assert spark.read.parquet(str(path)).first().asDict() == {
                "baseline": baseline,
                "report_name": report_name,
            }

    with pytest.raises(FileExistsError):
        save_evaluation_reports(results, paths)

    for baseline, report_paths in paths.items():
        for report_name, path in report_paths.items():
            assert spark.read.parquet(str(path)).first().asDict() == {
                "baseline": baseline,
                "report_name": report_name,
            }
