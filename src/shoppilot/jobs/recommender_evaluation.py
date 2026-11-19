from pathlib import Path
from shoppilot.recommendations.config import ModelInputConfig, calculate_config_hash
from shoppilot.recommendations.evaluation_pipeline import evaluate_category_popularity, evaluate_global_popularity


def load_evaluation_inputs(spark, config: ModelInputConfig, project_root=".") -> dict:
    """Load evaluation inputs from a config.

    Args:
        spark: Spark session.
        config: Model input configuration.
        project_root: Root directory of the project.

    Returns:
        A dictionary of evaluation inputs.
    """

    project_root = Path(project_root)
    gold_path = project_root / "data" / "lake" / "gold" / "snapshots" / config.snapshot_id
    model_inputs_path = project_root / "data" / "lake" / "model_inputs" / config.snapshot_id / calculate_config_hash(config) / "artifacts"
    assignments_df = spark.read.parquet(
        str(gold_path / "evaluation_assignments" / config.protocol_version)
    )
    product_documents_df = spark.read.parquet(
        str(gold_path / "product_documents")
    )
    candidate_universe_df = spark.read.parquet(str(model_inputs_path / "candidate_universe"))
    seen_items_df = spark.read.parquet(str(model_inputs_path / "seen_items"))
    item_popularity_df = spark.read.parquet(str(model_inputs_path / "item_popularity"))
    
    return {
        "assignments": assignments_df,
        "products": product_documents_df,
        "candidate_universe": candidate_universe_df,
        "seen_items": seen_items_df,
        "item_popularity": item_popularity_df,
    }


def build_evaluation_output_paths(config, k, project_root="."):
    """Build the output paths for the evaluation results.

    Args:
        config: Model input configuration.
        k: Length of the recommendation list.
        project_root: Root directory of the project.

    Returns:
        A dictionary of output paths.
    """

    project_root = Path(project_root)
    output_path = project_root / "data" / "lake" / "recommender_evaluations" / config.snapshot_id / calculate_config_hash(config) / f"k={k}"
    return {
        "global_popularity": {
            "scored_rows": output_path / "global_popularity" / "scored_rows",
            "overall_report": output_path / "global_popularity" / "overall_report",
            "user_availability_report": output_path / "global_popularity" / "user_availability_report",
        },
        "category_popularity": {
            "scored_rows": output_path / "category_popularity" / "scored_rows",
            "overall_report": output_path / "category_popularity" / "overall_report",
            "user_availability_report": output_path / "category_popularity" / "user_availability_report",
        }
    }


def save_evaluation_reports(results, paths):
    """Save the evaluation reports.

    Args:
        results: A dictionary of evaluation results.
        paths: A dictionary of output paths.
    """

    # если существует хотя бы один из файлов, то выбрасываем ошибку
    if any(report_path.exists() for baseline in paths.values() for report_path in baseline.values()):
        raise FileExistsError(f"One or more files already exist at the following paths: {paths}")

    for baseline_name in results:
        for report_name, report_path in paths[baseline_name].items():
            results[baseline_name][report_name].write.mode("errorifexists").parquet(str(report_path))


def run_recommender_evaluation_job(spark, config, k=20, project_root="."):
    """Run the recommender evaluation job.

    Args:
        spark: Spark session.
        config: Model input configuration.
        k: Length of the recommendation list.
        project_root: Root directory of the project.

    Returns:
        A dictionary of evaluation results.
        "global_popularity": Global popularity evaluation results.
        "category_popularity": Category popularity evaluation results.
    """

    evaluation_inputs = load_evaluation_inputs(spark, config, project_root)
    
    global_evaluation_results = evaluate_global_popularity(
        evaluation_inputs["assignments"], 
        evaluation_inputs["candidate_universe"], 
        evaluation_inputs["seen_items"], 
        evaluation_inputs["item_popularity"], 
        config, 
        k
    )

    category_evaluation_results = evaluate_category_popularity(
        evaluation_inputs["assignments"], 
        evaluation_inputs["candidate_universe"], 
        evaluation_inputs["seen_items"], 
        evaluation_inputs["item_popularity"], 
        evaluation_inputs["products"], 
        config, 
        k
    )

    results = {
        "global_popularity": global_evaluation_results,
        "category_popularity": category_evaluation_results,
    }

    output_paths = build_evaluation_output_paths(config, k, project_root)
    save_evaluation_reports(results, output_paths)
    return results