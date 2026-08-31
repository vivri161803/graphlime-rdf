"""Typed configuration contracts (plan §7) — the single source of truth.

Every experiment is described by an :class:`ExperimentConfig`; every result row
is a validated :class:`ExplanationRecord`; every training run writes a
:class:`RunManifest`; every checkpoint bundle embeds a
:class:`CheckpointManifest`. No magic numbers live outside this module and the
YAML files in ``configs/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

DatasetName = Literal["aifb", "mutag", "synthetic"]


class FeatureSpaceConfig(BaseModel):
    """Interpretable feature space built from graph predicates (plan §6)."""

    kind: Literal["predicate", "predicate_object"]
    directions: list[Literal["out", "in"]] = ["out"]
    binary: bool = True
    min_support: int = 5
    model_config = ConfigDict(frozen=True, extra="forbid")


class GraphLIMEConfig(BaseModel):
    """GraphLIME / HSIC Lasso hyperparameters."""

    hops: int = Field(2, ge=1, le=3)
    rho: float = Field(0.1, gt=0)
    kernel: Literal["rbf"] = "rbf"
    sigma: float | None = None  # None → median heuristic
    min_neighborhood: int = Field(10, ge=2)
    # Hub nodes can have k-hop neighbourhoods of thousands of nodes; the HSIC
    # Lasso design matrix has n² rows, so we subsample deterministically
    # (seeded by the node id, target always kept) above this cap.
    max_neighborhood: int = Field(200, ge=10)
    model_config = ConfigDict(frozen=True, extra="forbid")


class RGCNModelConfig(BaseModel):
    """Two-layer R-GCN with basis decomposition."""

    hidden_dim: int = Field(16, ge=2)
    num_bases: int = Field(30, ge=1)
    dropout: float = Field(0.0, ge=0.0, lt=1.0)
    model_config = ConfigDict(frozen=True, extra="forbid")


class TrainingConfig(BaseModel):
    """Full-batch training hyperparameters."""

    epochs: int = Field(50, ge=1)
    lr: float = Field(0.01, gt=0)
    weight_decay: float = Field(0.0005, ge=0)
    seeds: list[int] = [0, 1, 2, 3, 4]
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExperimentConfig(BaseModel):
    """Resolved, fully-specified experiment: one YAML file in ``configs/``."""

    dataset: DatasetName
    feature_space: FeatureSpaceConfig
    model: RGCNModelConfig = RGCNModelConfig()
    training: TrainingConfig = TrainingConfig()
    graphlime: GraphLIMEConfig = GraphLIMEConfig()
    model_config = ConfigDict(frozen=True, extra="forbid")

    def config_hash(self) -> str:
        """Stable short hash of the canonical JSON serialisation."""
        canon = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(canon.encode()).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        with open(path) as f:
            return cls.model_validate(yaml.safe_load(f))

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=True)


class SyntheticConfig(BaseModel):
    """Ground-truth generator parameters (plan M6).

    Label rule: class 1 ⟺ the entity has ``target_predicate``. Distractor
    predicates co-occur with the target with probability ``distractor_strength``
    (and appear on class-0 entities with 1 − strength) — correlated but not
    causal. ``noise_rate`` flips that fraction of labels.
    """

    num_entities: int = Field(400, ge=20)
    num_predicates: int = Field(10, ge=2)
    target_predicate: int = Field(7, ge=0)
    edge_prob: float = Field(0.5, gt=0, lt=1)
    class_balance: float = Field(0.5, gt=0, lt=1)
    noise_rate: float = Field(0.0, ge=0, lt=1)
    num_distractors: int = Field(0, ge=0)
    distractor_strength: float = Field(0.9, gt=0.5, lt=1)
    train_fraction: float = Field(0.8, gt=0, lt=1)
    seed: int = 0
    model_config = ConfigDict(frozen=True, extra="forbid")


class ExplanationRecord(BaseModel):
    """One validated row per explained node, written to ``results/*.jsonl``.

    ``top_features`` carries **human-readable predicate names** — an integer
    id where a name is expected is a bug (see CLAUDE.md).
    """

    dataset: str
    node_id: int
    true_label: int
    predicted_label: int
    predicted_prob: float
    neighborhood_size: int
    top_features: list[tuple[str, float]]  # (human-readable name, beta)
    sparsity: float
    seed: int
    config_hash: str
    git_commit: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class RefusalRecord(BaseModel):
    """One row per refused explanation — a finding, not an error (plan M5)."""

    dataset: str
    node_id: int
    reason: str
    neighborhood_size: int
    seed: int
    config_hash: str
    git_commit: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class FidelityRecord(BaseModel):
    """Fidelity± at one k for one node, with the random-k control (plan M7)."""

    dataset: str
    node_id: int
    k: int
    fidelity_plus: float
    fidelity_minus: float
    random_plus: float
    random_minus: float
    seed: int
    config_hash: str
    git_commit: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class StabilityRecord(BaseModel):
    """Jaccard@k of top-k feature sets across a varied dimension (plan M7)."""

    dataset: str
    node_id: int
    kind: Literal["seeds", "hops"]
    variant_a: str
    variant_b: str
    k: int
    jaccard: float
    config_hash: str
    git_commit: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class AgreementRecord(BaseModel):
    """Agreement between explainers / feature spaces at top-k (plan M7)."""

    dataset: str
    node_id: int
    kind: Literal["feature_space", "baseline"]
    k: int
    jaccard: float
    seed: int
    config_hash: str
    git_commit: str
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunManifest(BaseModel):
    """One per training run, stored under ``runs/``."""

    run_id: str  # <timestamp>_<config_hash>
    dataset: str
    seed: int
    git_commit: str
    resolved_config: ExperimentConfig
    library_versions: dict[str, str]
    final_test_accuracy: float
    epochs: int
    model_config = ConfigDict(frozen=True, extra="forbid")


class CheckpointManifest(BaseModel):
    """Embedded inside every self-contained ``.pt`` bundle (plan §9.2)."""

    dataset: str
    created_at: str
    git_commit: str
    seed: int
    resolved_config: ExperimentConfig
    library_versions: dict[str, str]
    test_accuracy: float
    vocabulary_hash: str
    model_config = ConfigDict(frozen=True, extra="forbid")


def library_versions() -> dict[str, str]:
    """Versions of the packages that determine numerical results."""
    import importlib.metadata as md

    return {
        pkg: md.version(pkg)
        for pkg in ["torch", "torch-geometric", "numpy", "scipy", "scikit-learn", "rdflib"]
    }


def current_git_commit() -> str:
    """Short hash of HEAD, linking every result row to an exact code state."""
    import subprocess

    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )
    return out.stdout.strip()
