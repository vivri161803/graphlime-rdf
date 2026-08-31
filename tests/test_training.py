"""M3 DoD tests: overfit gate, bit-determinism, checkpoint round-trip, audit trail."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    RGCNModelConfig,
    RunManifest,
    TrainingConfig,
)
from graphlime_rdf.data.synthetic import tiny_overfit_graph
from graphlime_rdf.training import (
    load_checkpoint,
    save_checkpoint,
    train_run,
    write_run_dir,
)

REPO = Path(__file__).resolve().parents[1]


def _tiny_config(epochs: int = 100) -> ExperimentConfig:
    return ExperimentConfig(
        dataset="synthetic",
        feature_space=FeatureSpaceConfig(kind="predicate", min_support=1),
        model=RGCNModelConfig(hidden_dim=8, num_bases=4),
        training=TrainingConfig(epochs=epochs, weight_decay=0.0, seeds=[0]),
    )


def test_overfits_tiny_synthetic_graph_to_100_percent() -> None:
    graph = tiny_overfit_graph()
    result = train_run(graph, _tiny_config(), seed=0)
    final_train_acc = result.history[-1][2]
    assert final_train_acc == 1.0


def test_same_seed_gives_bit_identical_logits() -> None:
    graph = tiny_overfit_graph()
    r1 = train_run(graph, _tiny_config(epochs=30), seed=7)
    r2 = train_run(graph, _tiny_config(epochs=30), seed=7)
    l1 = r1.model(r1.x, r1.edge_index, r1.edge_type)
    l2 = r2.model(r2.x, r2.edge_index, r2.edge_type)
    assert torch.equal(l1, l2)


def test_different_seeds_differ() -> None:
    graph = tiny_overfit_graph()
    r1 = train_run(graph, _tiny_config(epochs=30), seed=0)
    r2 = train_run(graph, _tiny_config(epochs=30), seed=1)
    l1 = r1.model(r1.x, r1.edge_index, r1.edge_type)
    l2 = r2.model(r2.x, r2.edge_index, r2.edge_type)
    assert not torch.equal(l1, l2)


def test_run_dir_audit_trail(tmp_path: Path) -> None:
    graph = tiny_overfit_graph()
    result = train_run(graph, _tiny_config(epochs=5), seed=0)
    run_dir = write_run_dir(result, tmp_path)
    manifest = RunManifest.model_validate_json((run_dir / "manifest.json").read_text())
    assert manifest.dataset == "synthetic"
    assert manifest.epochs == 5
    assert ExperimentConfig.from_yaml(run_dir / "config.yaml") == result.manifest.resolved_config
    lines = (run_dir / "epochs.csv").read_text().splitlines()
    assert lines[0] == "epoch,train_loss,train_acc,test_acc"
    assert len(lines) == 6


def test_checkpoint_round_trip_in_process(tmp_path: Path) -> None:
    graph = tiny_overfit_graph()
    result = train_run(graph, _tiny_config(epochs=30), seed=0)
    path = tmp_path / "tiny_best.pt"
    save_checkpoint(result, graph, path)

    loaded = load_checkpoint(path)
    assert loaded.manifest.dataset == "synthetic"
    assert loaded.vocabulary == result.vocabulary
    assert loaded.label_map == {"class_0": 0, "class_1": 1}
    original = result.model(result.x, result.edge_index, result.edge_type)
    reloaded = loaded.model(result.x, result.edge_index, result.edge_type)
    assert torch.equal(original, reloaded)


@pytest.mark.slow
def test_checkpoint_round_trip_fresh_process(tmp_path: Path) -> None:
    graph = tiny_overfit_graph()
    result = train_run(graph, _tiny_config(epochs=30), seed=0)
    path = tmp_path / "tiny_best.pt"
    save_checkpoint(result, graph, path)

    result.model.eval()
    with torch.no_grad():
        logits = result.model(result.x, result.edge_index, result.edge_type)
    parent_digest = hashlib.sha256(logits.numpy().tobytes()).hexdigest()

    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "roundtrip_child.py"), str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    assert out.stdout.strip() == parent_digest


def _mean_accuracy(dataset: str) -> float:
    import statistics

    from graphlime_rdf.data.loader import load_rdf_graph

    cfg = ExperimentConfig.from_yaml(REPO / "configs" / f"{dataset}.yaml")
    graph = load_rdf_graph(dataset)
    accs = [
        train_run(graph, cfg, seed).manifest.final_test_accuracy
        for seed in cfg.training.seeds
    ]
    print(f"{dataset} per-seed accuracies: {json.dumps(accs)}")
    return statistics.mean(accs)


@pytest.mark.slow
def test_aifb_accuracy_meets_dod() -> None:
    assert _mean_accuracy("aifb") >= 0.85


@pytest.mark.slow
def test_mutag_accuracy_meets_dod() -> None:
    assert _mean_accuracy("mutag") >= 0.62
