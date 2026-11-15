from dataclasses import fields, replace

from shoppilot.recommendations.config import (
    ModelInputConfig,
    calculate_config_hash,
)


def make_config():
    return ModelInputConfig(
        snapshot_id="snapshot-v1",
        signal_version="signal-v1",
        protocol_version="protocol-v1",
        cutoff="2021-01-01",
    )


def test_model_input_config_is_immutable():
    config = make_config()

    try:
        config.cutoff = "2022-01-01"
    except AttributeError:
        pass
    else:
        raise AssertionError("ModelInputConfig must be immutable")


def test_config_hash_is_canonical_and_stable():
    config = make_config()

    assert calculate_config_hash(config) == (
        "9fce9850cecf0930d47d5c1c79c58ad5d386fbf9150d362c0bc9ac1a82b4278e"
    )
    assert calculate_config_hash(make_config()) == calculate_config_hash(config)


def test_every_config_field_contributes_to_hash():
    config = make_config()
    original_hash = calculate_config_hash(config)

    replacements = {
        "snapshot_id": "snapshot-v2",
        "signal_version": "signal-v2",
        "protocol_version": "protocol-v2",
        "cutoff": "2022-01-01",
        "aggregation_version": "binary-unique-pair-v2",
        "candidate_policy_version": "positive-train-items-v2",
        "popularity_policy_version": "positive-pair-count-v2",
        "training_role": "retrain",
    }

    assert set(replacements) == {field.name for field in fields(config)}
    for name, value in replacements.items():
        assert calculate_config_hash(replace(config, **{name: value})) != original_hash
