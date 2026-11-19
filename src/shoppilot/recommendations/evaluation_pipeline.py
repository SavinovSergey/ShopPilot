"""Orchestrate train-only recommendation baselines and validation evaluation."""

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.evaluation import (build_validation_targets, 
    build_target_eligibility, build_evaluation_rows, score_evaluation_rows, 
    summarize_evaluation_rows)
from shoppilot.recommendations.popularity import (rank_global_popularity,
    recommend_global_popularity, rank_category_popularity, build_user_category_preferences,
    build_user_category_candidates, recommend_from_category_candidates, fill_category_recommendations_with_global)
from shoppilot.recommendations.diagnostics import calculate_catalog_coverage, calculate_novelty_at_k
from shoppilot.recommendations.evaluation_report import add_evaluation_provenance


def evaluate_category_popularity(
    assignments_df,
    candidate_universe_df,
    seen_items_df,
    item_popularity_df,
    products_df,    
    config: ModelInputConfig,
    k: int,
) -> dict:
    """Evaluate category popularity on validation targets for one configuration.

    Args:
        assignments_df: Versioned Gold evaluation assignments.
        candidate_universe_df: Versioned train-known products.
        seen_items_df: Versioned train-seen user-product pairs.
        item_popularity_df: Versioned train-only product popularity counts.
        products_df: Current product catalog with categories. Category-based
            results are retrospective because the metadata is not guaranteed
            to reflect product state at the train cutoff.
        config: Snapshot, signal, protocol, cutoff, and aggregation versions.
        k: Positive length of each recommendation list and metric cutoff.

    Returns:
        A dictionary of Spark DataFrames:
        ``scored_rows`` — one row per target user;
        ``overall_report`` — one row per tenant with quality metrics,
        target counts, catalog coverage at K and novelty at K;
        ``user_availability_report`` — one row per tenant and user segment
        with quality metrics and target counts.
        Users with no eligible targets remain in the reports.
    """

    validation_targets_df = build_validation_targets(assignments_df, config)
    users_df = validation_targets_df.select("tenant_id", "user_id")

    # filter seen_items_df to only include rows for users in validation_targets_df
    seen_items_df = seen_items_df.join(
        users_df,
        on = ["tenant_id", "user_id"],
        how = "left_semi"
    )

    target_eligibility_df = build_target_eligibility(
        validation_targets_df, 
        candidate_universe_df, 
        seen_items_df,
        config
    )
    category_ranked_df = rank_category_popularity(item_popularity_df, products_df, config)
    category_preferences_df = build_user_category_preferences(seen_items_df, products_df, config)
    category_candidates_df = build_user_category_candidates(
        category_preferences_df, 
        category_ranked_df, 
        seen_items_df, 
        config,
        k
    )
    category_recommendations_df = recommend_from_category_candidates(users_df, category_candidates_df, k)
    global_ranked_df = rank_global_popularity(item_popularity_df, config)
    
    category_recommendations_with_global_df = fill_category_recommendations_with_global(
        users_df, 
        category_recommendations_df, 
        global_ranked_df, 
        seen_items_df, 
        config, 
        k
    )

    evaluation_rows_df = build_evaluation_rows(
        target_eligibility_df, 
        category_recommendations_with_global_df
    )
    scored_rows_df = score_evaluation_rows(evaluation_rows_df, k)
    catalog_coverage_df = calculate_catalog_coverage(scored_rows_df, candidate_universe_df, config, k)
    novelty_df = calculate_novelty_at_k(scored_rows_df, item_popularity_df, config, k)
    overall_report_df = summarize_evaluation_rows(scored_rows_df)
    overall_report_df = overall_report_df.join(
        catalog_coverage_df, 
        on="tenant_id", 
        how="left"
    )
    overall_report_df = overall_report_df.join(
        novelty_df,
        on="tenant_id",
        how="left"
    )
    user_availability_report_df = summarize_evaluation_rows(
        scored_rows_df, 
        slice_columns=("user_availability_state",)
    )

    scored_rows_df = add_evaluation_provenance(scored_rows_df, config, "category_popularity", k)
    overall_report_df = add_evaluation_provenance(overall_report_df, config, "category_popularity", k)
    user_availability_report_df = add_evaluation_provenance(user_availability_report_df, config, "category_popularity", k)
    return {
        "scored_rows": scored_rows_df,
        "overall_report": overall_report_df,
        "user_availability_report": user_availability_report_df
    }



def evaluate_global_popularity(
    assignments_df,
    candidate_universe_df,
    seen_items_df,
    item_popularity_df,
    config: ModelInputConfig,
    k: int,
) -> dict:
    """Evaluate global popularity on validation targets for one configuration.

    Args:
        assignments_df: Versioned Gold evaluation assignments.
        candidate_universe_df: Versioned train-known products.
        seen_items_df: Versioned train-seen user-product pairs.
        item_popularity_df: Versioned train-only product popularity counts.
        config: Snapshot, signal, protocol, cutoff, and aggregation versions.
        k: Positive length of each recommendation list and metric cutoff.

    Returns:
        A dictionary of Spark DataFrames:
        ``scored_rows`` — one row per target user;
        ``overall_report`` — one row per tenant with quality metrics,
        target counts, catalog coverage at K and novelty at K;
        ``user_availability_report`` — one row per tenant and user segment
        with quality metrics and target counts.
        Users with no eligible targets remain in the reports.
    """

    validation_targets_df = build_validation_targets(assignments_df, config)
    target_eligibility_df = build_target_eligibility(
        validation_targets_df, 
        candidate_universe_df, 
        seen_items_df, 
        config
    )
    users_df = validation_targets_df.select("tenant_id", "user_id")
    ranked_df = rank_global_popularity(item_popularity_df, config)
    recommendations_df = recommend_global_popularity(users_df, ranked_df, seen_items_df, config, k)
    evaluation_rows_df = build_evaluation_rows(target_eligibility_df, recommendations_df)
    scored_rows_df = score_evaluation_rows(evaluation_rows_df, k)
    catalog_coverage_df = calculate_catalog_coverage(scored_rows_df, candidate_universe_df, config, k)
    novelty_df = calculate_novelty_at_k(scored_rows_df, item_popularity_df, config, k)
    overall_report_df = summarize_evaluation_rows(scored_rows_df)
    overall_report_df = overall_report_df.join(
        catalog_coverage_df, 
        on="tenant_id", 
        how="left"
    )
    overall_report_df = overall_report_df.join(
        novelty_df,
        on="tenant_id",
        how="left"
    )
    user_availability_report_df = summarize_evaluation_rows(
        scored_rows_df, 
        slice_columns=("user_availability_state",)
    )
    
    scored_rows_df = add_evaluation_provenance(scored_rows_df, config, "global_popularity", k)
    overall_report_df = add_evaluation_provenance(overall_report_df, config, "global_popularity", k)
    user_availability_report_df = add_evaluation_provenance(user_availability_report_df, config, "global_popularity", k)
    return {
        "scored_rows": scored_rows_df,
        "overall_report": overall_report_df,
        "user_availability_report": user_availability_report_df
    }
