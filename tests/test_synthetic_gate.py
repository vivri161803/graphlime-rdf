"""M6 HARD GATE: GraphLIME must recover the planted ground truth (plan M6).

Nothing on AIFB/MUTAG is reported until this file passes. Thresholds are part
of the project contract and are never lowered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from graphlime_rdf.config import ExperimentConfig, SyntheticConfig
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.explain.graphlime import Explanation, explain_node
from graphlime_rdf.training import TrainResult, train_run

REPO = Path(__file__).resolve().parents[1]
TARGET_FEATURE = "out:syn:p7"


def _trained(syn: SyntheticConfig) -> tuple[TrainResult, RDFGraph]:
    graph = generate_ground_truth_graph(syn)
    cfg = ExperimentConfig.from_yaml(REPO / "configs" / "synthetic.yaml")
    result = train_run(graph, cfg, seed=0)
    # Sanity precondition (not the gate): the rule is learnable by design.
    assert result.manifest.final_test_accuracy >= 0.9, (
        f"model failed to learn the planted rule: "
        f"acc={result.manifest.final_test_accuracy:.3f}"
    )
    return result, graph


def _rank_of_target(explanation: Explanation) -> int:
    """1-based rank of the ground-truth feature within the β ordering."""
    names = [name for name, _ in explanation.top_features(len(explanation.beta))]
    if TARGET_FEATURE not in names:
        return len(explanation.beta) + 1
    return names.index(TARGET_FEATURE) + 1


def _gate_ranks(result: TrainResult, graph: RDFGraph) -> list[int]:
    cfg = ExperimentConfig.from_yaml(REPO / "configs" / "synthetic.yaml")
    ranks = []
    test_nodes = graph.test_mask.nonzero().flatten().tolist()
    for node in test_nodes:
        out = explain_node(
            result.model, int(node), result.x, result.edge_index, result.edge_type,
            result.vocabulary, cfg.graphlime,
        )
        if isinstance(out, Explanation):
            ranks.append(_rank_of_target(out))
    assert len(ranks) >= 0.8 * len(test_nodes), "too many refusals to run the gate"
    return ranks


@pytest.mark.slow
def test_gate_zero_noise_target_ranked_first() -> None:
    """With 0% noise, the planted predicate is ranked FIRST for ≥95% of nodes."""
    result, graph = _trained(SyntheticConfig())
    ranks = _gate_ranks(result, graph)
    top1_rate = sum(r == 1 for r in ranks) / len(ranks)
    assert top1_rate >= 0.95, f"top-1 rate {top1_rate:.3f} over {len(ranks)} nodes"


@pytest.mark.slow
def test_gate_correlated_distractors_target_stays_top3() -> None:
    """With strong correlated distractors, the planted predicate stays top-3."""
    result, graph = _trained(SyntheticConfig(num_distractors=2, distractor_strength=0.9))
    ranks = _gate_ranks(result, graph)
    top3_rate = sum(r <= 3 for r in ranks) / len(ranks)
    assert top3_rate >= 0.95, f"top-3 rate {top3_rate:.3f} over {len(ranks)} nodes"
