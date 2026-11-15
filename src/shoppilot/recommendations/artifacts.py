"""Reproducibility metadata for recommender model-input artifacts."""
from shoppilot.recommendations.model_input import MAPPING_SCOPE_COLUMNS
from shoppilot.data.signatures import calculate_dataset_signature


MODEL_INPUT_ARTIFACTS = {
    "train_pairs": {
        "business_keys": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "user_id", "product_id"
        ),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "user_id", "product_id",
            "event_count", "preference","confidence"
        ),
    },
    "user_mapping": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "user_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "user_id", "user_index"
        ),
    },
    "item_mapping": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "product_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "product_id", "item_index"
        ),
    },
    "indexed_pairs": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "user_id", "product_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "user_id", "product_id",
            "event_count", "preference", "confidence",
            "user_index", "item_index"
        ),
    },
    "seen_items": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "user_id", "product_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "user_id", "product_id",
            "user_index", "item_index"
        ),
    },
    "candidate_universe": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "product_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "product_id", "item_index"
        ),
    },
    "item_popularity": {
        "business_keys": (*MAPPING_SCOPE_COLUMNS, "tenant_id", "product_id"),
        "checksum_columns": (
            *MAPPING_SCOPE_COLUMNS,
            "tenant_id", "product_id", "item_index", "positive_pair_count"
        ),
    },
}

MODEL_INPUT_ARTIFACT_NAMES = tuple(MODEL_INPUT_ARTIFACTS.keys())


def calculate_model_input_signatures(datasets: dict) -> dict:
    """Return logical dataset signatures for all model-input artifacts."""
    input_signatures = {}
    for artifact_name, artifact_config in MODEL_INPUT_ARTIFACTS.items():
        if artifact_name not in datasets:
            raise ValueError(f"Dataset {artifact_name} not found in input datasets")
        business_keys = artifact_config["business_keys"]
        checksum_columns = artifact_config["checksum_columns"]
        signature = calculate_dataset_signature(
            datasets[artifact_name],
            business_keys,
            checksum_columns,
        )
        input_signatures[artifact_name] = signature
    return input_signatures
