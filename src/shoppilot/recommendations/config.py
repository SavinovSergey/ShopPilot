"""Versioned configuration for reproducible recommender model inputs."""

from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class ModelInputConfig:
    """All semantic choices which identify one model-input build."""

    snapshot_id: str
    signal_version: str
    protocol_version: str
    cutoff: str
    aggregation_version: str = "binary_unique_pair_v1"
    candidate_policy_version: str = "positive_train_items_v1"
    popularity_policy_version: str = "positive_pair_count_v1"
    training_role: str = "train"


def calculate_config_hash(config: ModelInputConfig) -> str:
    """Return a stable SHA-256 fingerprint of every semantic config field."""
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def validate_model_input_config(config: ModelInputConfig) -> None:
    """Reject invalid model-input configuration."""
    valid_config = {
        "training_role": ["train"],
        "aggregation_version": ["binary_unique_pair_v1"],
        "candidate_policy_version": ["positive_train_items_v1"],
        "popularity_policy_version": ["positive_pair_count_v1"]
    }
    for field, valid_values in valid_config.items():
        if getattr(config, field) not in valid_values:
            raise ValueError(f"Invalid {field}: {getattr(config, field)}")

