"""Config contract tests: hashing stability, YAML round-trip, validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    GraphLIMEConfig,
    library_versions,
)


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="aifb",
        feature_space=FeatureSpaceConfig(kind="predicate"),
    )


def test_config_hash_stable_and_sensitive() -> None:
    assert _config().config_hash() == _config().config_hash()
    other = ExperimentConfig(
        dataset="mutag", feature_space=FeatureSpaceConfig(kind="predicate")
    )
    assert other.config_hash() != _config().config_hash()


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = _config()
    path = tmp_path / "cfg.yaml"
    cfg.to_yaml(path)
    assert ExperimentConfig.from_yaml(path) == cfg


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        FeatureSpaceConfig(kind="predicate", surprise=1)  # type: ignore[call-arg]


def test_graphlime_bounds_enforced() -> None:
    with pytest.raises(ValidationError):
        GraphLIMEConfig(hops=5)
    with pytest.raises(ValidationError):
        GraphLIMEConfig(rho=0.0)


def test_library_versions_cover_numerical_stack() -> None:
    versions = library_versions()
    assert {"torch", "numpy", "scikit-learn"} <= versions.keys()
    assert all(v for v in versions.values())
