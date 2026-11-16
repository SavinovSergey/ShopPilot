"""Orchestration for building reproducible recommender model inputs."""

from shoppilot.recommendations.config import ModelInputConfig
from shoppilot.recommendations.model_input import (
    select_training_events, aggregate_binary_pairs,
    build_seen_items, build_candidate_universe, build_item_popularity_counts,
    build_user_mapping, build_item_mapping, attach_mapping_indices,
    calculate_mapping_coverage_metrics, validate_mapping_coverage_metrics
)
from shoppilot.recommendations.quality import (
    calculate_training_leakage_metrics, validate_training_leakage_metrics
)
from shoppilot.recommendations.artifacts import calculate_model_input_signatures
from shoppilot.recommendations.config import calculate_config_hash, validate_model_input_config


def build_model_input_bundle(
    signals_df,
    assignments_df,
    config: ModelInputConfig,
) -> dict:
    """Build validated model-input artifacts and reproducibility metadata."""

    validate_model_input_config(config)

    training_events_df = select_training_events(
        signals_df,
        assignments_df,
        snapshot_id=config.snapshot_id,
        signal_version=config.signal_version,
        protocol_version=config.protocol_version,
        cutoff=config.cutoff
    )

    leakage_metrics = calculate_training_leakage_metrics(
        training_events_df,
        assignments_df,
        config
    )
    validate_training_leakage_metrics(leakage_metrics)

    train_pairs = aggregate_binary_pairs(
        training_events_df,
        config.aggregation_version
    )
    user_mapping = build_user_mapping(train_pairs)
    item_mapping = build_item_mapping(train_pairs)
    indexed_pairs = attach_mapping_indices(train_pairs, user_mapping, item_mapping)
    mapping_coverage_metrics = calculate_mapping_coverage_metrics(
        train_pairs,
        user_mapping,
        item_mapping,
        indexed_pairs
    )
    validate_mapping_coverage_metrics(mapping_coverage_metrics)

    seen_items = build_seen_items(indexed_pairs)
    candidate_universe = build_candidate_universe(indexed_pairs)
    item_popularity = build_item_popularity_counts(indexed_pairs)

    artifacts = {
        "train_pairs": train_pairs,
        "user_mapping": user_mapping,
        "item_mapping": item_mapping,
        "indexed_pairs": indexed_pairs,
        "seen_items": seen_items,
        "candidate_universe": candidate_universe,
        "item_popularity": item_popularity
    }

    signatures = calculate_model_input_signatures(artifacts)
    config_hash = calculate_config_hash(config)

    return {
        "artifacts": artifacts,
        "quality_metrics": {
            "leakage": leakage_metrics,
            "mapping_coverage": mapping_coverage_metrics
        },
        "signatures": signatures,
        "config_hash": config_hash
    }
