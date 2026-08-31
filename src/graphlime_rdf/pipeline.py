"""Experiment pipeline: explanations + evaluation → validated JSONL (plan M7).

Every row written to ``results/`` is a validated pydantic record carrying
``config_hash``, ``seed`` and ``git_commit`` — the full repeatability link.
All node sets are the datasets' *test* nodes, in sorted order; everything is
deterministic given the config.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from graphlime_rdf.config import (
    AgreementRecord,
    ExperimentConfig,
    ExplanationRecord,
    FidelityRecord,
    RefusalRecord,
    StabilityRecord,
    current_git_commit,
)
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.evaluate.agreement import agreement_at_k, graphlime_predicate_ranking
from graphlime_rdf.evaluate.fidelity import fidelity_at_k
from graphlime_rdf.evaluate.stability import jaccard_at_k
from graphlime_rdf.explain.baseline import gnnexplainer_predicate_ranking
from graphlime_rdf.explain.graphlime import Explanation, Refusal, explain_node
from graphlime_rdf.features import build_features
from graphlime_rdf.training import TrainResult, train_run

FIDELITY_KS = [1, 2, 5, 10]
STABILITY_K = 5
AGREEMENT_K = 5
TOP_FEATURES_RECORDED = 10

def write_jsonl[RecordT: BaseModel](records: list[RecordT], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")


def read_jsonl[RecordT: BaseModel](path: Path, model: type[RecordT]) -> list[RecordT]:
    return [
        model.model_validate(json.loads(line))
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def test_nodes_of(graph: RDFGraph) -> list[int]:
    return sorted(int(i) for i in graph.test_mask.nonzero().flatten())


def _explain_all(
    result: TrainResult, graph: RDFGraph, config: ExperimentConfig
) -> dict[int, Explanation | Refusal]:
    return {
        node: explain_node(
            result.model, node, result.x, result.edge_index, result.edge_type,
            result.vocabulary, config.graphlime,
        )
        for node in test_nodes_of(graph)
    }


def explanation_records(
    result: TrainResult, graph: RDFGraph, config: ExperimentConfig
) -> tuple[list[ExplanationRecord], list[RefusalRecord]]:
    """ExplanationRecord per explained test node; RefusalRecord per refusal."""
    commit = current_git_commit()
    chash = config.config_hash()
    seed = result.manifest.seed
    probs = result.model.predict_proba(result.x, result.edge_index, result.edge_type)

    explanations: list[ExplanationRecord] = []
    refusals: list[RefusalRecord] = []
    for node, out in _explain_all(result, graph, config).items():
        if isinstance(out, Refusal):
            refusals.append(
                RefusalRecord(
                    dataset=graph.dataset,
                    node_id=node,
                    reason=out.reason,
                    neighborhood_size=out.neighborhood_size,
                    seed=seed,
                    config_hash=chash,
                    git_commit=commit,
                )
            )
            continue
        predicted = int(probs[node].argmax())
        explanations.append(
            ExplanationRecord(
                dataset=graph.dataset,
                node_id=node,
                true_label=int(graph.labels[node]),
                predicted_label=predicted,
                predicted_prob=float(probs[node, predicted]),
                neighborhood_size=out.neighborhood_size,
                top_features=out.top_features(TOP_FEATURES_RECORDED),
                sparsity=out.sparsity,
                seed=seed,
                config_hash=chash,
                git_commit=commit,
            )
        )
    return explanations, refusals


def fidelity_records(
    result: TrainResult, graph: RDFGraph, config: ExperimentConfig
) -> list[FidelityRecord]:
    """Fidelity± with random-k control, swept over k, per explained test node."""
    commit = current_git_commit()
    chash = config.config_hash()
    records = []
    for node, out in _explain_all(result, graph, config).items():
        if not isinstance(out, Explanation):
            continue
        ranked = [int(j) for j in np.argsort(-out.beta, kind="stable")]
        for k in FIDELITY_KS:
            fid_plus, fid_minus, rand_plus, rand_minus = fidelity_at_k(
                result.model, result.x, result.edge_index, result.edge_type,
                node, ranked, k,
            )
            records.append(
                FidelityRecord(
                    dataset=graph.dataset,
                    node_id=node,
                    k=k,
                    fidelity_plus=fid_plus,
                    fidelity_minus=fid_minus,
                    random_plus=rand_plus,
                    random_minus=rand_minus,
                    seed=result.manifest.seed,
                    config_hash=chash,
                    git_commit=commit,
                )
            )
    return records


def stability_records(
    results_by_seed: dict[int, TrainResult], graph: RDFGraph, config: ExperimentConfig
) -> list[StabilityRecord]:
    """Jaccard@k across training seeds and across hops ∈ {1,2,3}."""
    commit = current_git_commit()
    chash = config.config_hash()
    records = []

    # Across seeds: same node, models from different seeds.
    per_seed: dict[int, dict[int, Explanation | Refusal]] = {
        seed: _explain_all(result, graph, config)
        for seed, result in sorted(results_by_seed.items())
    }
    seeds = sorted(per_seed)
    for node in test_nodes_of(graph):
        for a, b in pairwise(seeds):
            ea, eb = per_seed[a][node], per_seed[b][node]
            if isinstance(ea, Explanation) and isinstance(eb, Explanation):
                records.append(
                    StabilityRecord(
                        dataset=graph.dataset,
                        node_id=node,
                        kind="seeds",
                        variant_a=f"seed={a}",
                        variant_b=f"seed={b}",
                        k=STABILITY_K,
                        jaccard=jaccard_at_k(ea, eb, STABILITY_K),
                        config_hash=chash,
                        git_commit=commit,
                    )
                )

    # Across hops: best-seed model, varying the neighbourhood radius.
    best_seed = max(
        results_by_seed, key=lambda s: results_by_seed[s].manifest.final_test_accuracy
    )
    best = results_by_seed[best_seed]
    per_hops: dict[int, dict[int, Explanation | Refusal]] = {}
    for hops in [1, 2, 3]:
        gl = config.graphlime.model_copy(update={"hops": hops})
        per_hops[hops] = {
            node: explain_node(
                best.model, node, best.x, best.edge_index, best.edge_type,
                best.vocabulary, gl,
            )
            for node in test_nodes_of(graph)
        }
    for node in test_nodes_of(graph):
        for h1, h2 in [(1, 2), (2, 3)]:
            ea, eb = per_hops[h1][node], per_hops[h2][node]
            if isinstance(ea, Explanation) and isinstance(eb, Explanation):
                records.append(
                    StabilityRecord(
                        dataset=graph.dataset,
                        node_id=node,
                        kind="hops",
                        variant_a=f"hops={h1}",
                        variant_b=f"hops={h2}",
                        k=STABILITY_K,
                        jaccard=jaccard_at_k(ea, eb, STABILITY_K),
                        config_hash=chash,
                        git_commit=commit,
                    )
                )
    return records


def agreement_records(
    result: TrainResult,
    graph: RDFGraph,
    config: ExperimentConfig,
    baseline_epochs: int = 100,
) -> list[AgreementRecord]:
    """A-vs-B feature-space sensitivity and GraphLIME-vs-GNNExplainer agreement."""
    commit = current_git_commit()
    chash = config.config_hash()
    seed = result.manifest.seed
    records = []

    space_b = config.feature_space.model_copy(update={"kind": "predicate_object"})
    xb, vocab_b = build_features(graph, space_b)

    for node in test_nodes_of(graph):
        out_a = explain_node(
            result.model, node, result.x, result.edge_index, result.edge_type,
            result.vocabulary, config.graphlime,
        )
        if not isinstance(out_a, Explanation):
            continue
        ranking_a = graphlime_predicate_ranking(out_a)

        out_b = explain_node(
            result.model, node, result.x, result.edge_index, result.edge_type,
            vocab_b, config.graphlime, interpretable_x=xb,
        )
        if isinstance(out_b, Explanation):
            records.append(
                AgreementRecord(
                    dataset=graph.dataset,
                    node_id=node,
                    kind="feature_space",
                    k=AGREEMENT_K,
                    jaccard=agreement_at_k(
                        ranking_a, graphlime_predicate_ranking(out_b), AGREEMENT_K
                    ),
                    seed=seed,
                    config_hash=chash,
                    git_commit=commit,
                )
            )

        baseline = gnnexplainer_predicate_ranking(
            result.model, result.x, result.edge_index, result.edge_type,
            result.vocabulary, node, epochs=baseline_epochs, seed=seed,
        )
        records.append(
            AgreementRecord(
                dataset=graph.dataset,
                node_id=node,
                kind="baseline",
                k=AGREEMENT_K,
                jaccard=agreement_at_k(ranking_a, baseline, AGREEMENT_K),
                seed=seed,
                config_hash=chash,
                git_commit=commit,
            )
        )
    return records


def run_full_evaluation(config: ExperimentConfig, results_dir: Path = Path("results")) -> None:
    """Train all seeds, then write every record family for one dataset."""
    from graphlime_rdf.data.loader import load_rdf_graph

    graph = load_rdf_graph(config.dataset)
    results_by_seed = {
        seed: train_run(graph, config, seed) for seed in config.training.seeds
    }
    best_seed = max(
        results_by_seed, key=lambda s: results_by_seed[s].manifest.final_test_accuracy
    )
    best = results_by_seed[best_seed]
    ds = config.dataset

    explanations, refusals = explanation_records(best, graph, config)
    write_jsonl(explanations, results_dir / f"explanations_{ds}.jsonl")
    write_jsonl(refusals, results_dir / f"refusals_{ds}.jsonl")
    write_jsonl(
        fidelity_records(best, graph, config), results_dir / f"fidelity_{ds}.jsonl"
    )
    write_jsonl(
        stability_records(results_by_seed, graph, config),
        results_dir / f"stability_{ds}.jsonl",
    )
    write_jsonl(
        agreement_records(best, graph, config), results_dir / f"agreement_{ds}.jsonl"
    )
