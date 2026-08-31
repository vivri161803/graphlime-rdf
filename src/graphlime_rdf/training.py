"""Deterministic full-batch training with an auditable trail (plan M3).

Every run writes ``runs/<scratch|final>/<run_id>/`` containing a validated
:class:`RunManifest` (JSON), the resolved config (YAML) and ``epochs.csv``.
Best checkpoints are saved as self-contained bundles (plan §9.2).
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from graphlime_rdf.config import (
    CheckpointManifest,
    ExperimentConfig,
    RunManifest,
    current_git_commit,
    library_versions,
)
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.features import build_features
from graphlime_rdf.features.vocabulary import Vocabulary
from graphlime_rdf.models.rgcn import RGCN


def seed_everything(seed: int) -> None:
    """Fix all RNGs and force deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


@dataclass(frozen=True)
class TrainResult:
    """Outcome of one seeded training run."""

    model: RGCN
    manifest: RunManifest
    x: Tensor
    vocabulary: Vocabulary
    edge_index: Tensor
    edge_type: Tensor
    history: list[tuple[int, float, float, float]]  # epoch, loss, train_acc, test_acc


def _accuracy(logits: Tensor, labels: Tensor, mask: Tensor) -> float:
    pred = logits[mask].argmax(dim=-1)
    return float((pred == labels[mask]).float().mean())


def train_run(graph: RDFGraph, config: ExperimentConfig, seed: int) -> TrainResult:
    """Train one seeded model; returns everything needed to audit and reuse it."""
    seed_everything(seed)
    x, vocabulary = build_features(graph, config.feature_space)
    edge_index, edge_type = graph.doubled_edges()
    num_relations = 2 * graph.num_relations

    model = RGCN(x.shape[1], graph.num_classes, num_relations, config.model)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )

    history: list[tuple[int, float, float, float]] = []
    for epoch in range(1, config.training.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index, edge_type)
        loss = torch.nn.functional.cross_entropy(
            logits[graph.train_mask], graph.labels[graph.train_mask]
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        model.eval()
        with torch.no_grad():
            eval_logits = model(x, edge_index, edge_type)
        history.append(
            (
                epoch,
                float(loss.detach()),
                _accuracy(eval_logits, graph.labels, graph.train_mask),
                _accuracy(eval_logits, graph.labels, graph.test_mask),
            )
        )

    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}_{config.config_hash()}_s{seed}"
    manifest = RunManifest(
        run_id=run_id,
        dataset=graph.dataset,
        seed=seed,
        git_commit=current_git_commit(),
        resolved_config=config,
        library_versions=library_versions(),
        final_test_accuracy=history[-1][3],
        epochs=config.training.epochs,
    )
    return TrainResult(
        model=model,
        manifest=manifest,
        x=x,
        vocabulary=vocabulary,
        edge_index=edge_index,
        edge_type=edge_type,
        history=history,
    )


def write_run_dir(result: TrainResult, base: str | Path) -> Path:
    """Persist the audit trail: manifest JSON, resolved config YAML, epochs.csv."""
    run_dir = Path(base) / result.manifest.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(result.manifest.model_dump_json(indent=2))
    result.manifest.resolved_config.to_yaml(run_dir / "config.yaml")
    with open(run_dir / "epochs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_acc"])
        writer.writerows(result.history)
    return run_dir


def save_checkpoint(result: TrainResult, graph: RDFGraph, path: str | Path) -> None:
    """Write the self-contained bundle of plan §9.2."""
    manifest = CheckpointManifest(
        dataset=graph.dataset,
        created_at=datetime.now(UTC).isoformat(),
        git_commit=result.manifest.git_commit,
        seed=result.manifest.seed,
        resolved_config=result.manifest.resolved_config,
        library_versions=result.manifest.library_versions,
        test_accuracy=result.manifest.final_test_accuracy,
        vocabulary_hash=result.vocabulary.hash,
    )
    bundle = {
        "state_dict": result.model.state_dict(),
        "manifest": manifest.model_dump(mode="json"),
        "vocabulary": list(result.vocabulary.names),
        "label_map": {name: i for i, name in enumerate(graph.label_names)},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


@dataclass(frozen=True)
class LoadedCheckpoint:
    """A reconstructed model plus everything needed to use it standalone."""

    model: RGCN
    manifest: CheckpointManifest
    vocabulary: Vocabulary
    label_map: dict[str, int]


def load_checkpoint(path: str | Path) -> LoadedCheckpoint:
    """Rebuild the model from a bundle alone — no retraining, no external config."""
    bundle = torch.load(path, weights_only=True)
    manifest = CheckpointManifest.model_validate(bundle["manifest"])
    vocabulary = Vocabulary(names=tuple(bundle["vocabulary"]))
    if vocabulary.hash != manifest.vocabulary_hash:
        raise RuntimeError(
            f"vocabulary hash mismatch: bundle={vocabulary.hash}, "
            f"manifest={manifest.vocabulary_hash}"
        )
    label_map: dict[str, int] = dict(bundle["label_map"])
    state = bundle["state_dict"]
    num_relations = int(state["conv1.comp"].shape[0])
    model = RGCN(len(vocabulary), len(label_map), num_relations, manifest.resolved_config.model)
    model.load_state_dict(state)
    model.eval()
    return LoadedCheckpoint(
        model=model, manifest=manifest, vocabulary=vocabulary, label_map=label_map
    )
