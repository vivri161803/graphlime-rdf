"""M7 tests: fidelity masking semantics, stability metrics, agreement mapping,
GNNExplainer baseline integration (edge_type forwarding)."""

from __future__ import annotations

import numpy as np
import pytest

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    RGCNModelConfig,
    SyntheticConfig,
    TrainingConfig,
)
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.data.synthetic import generate_ground_truth_graph, tiny_overfit_graph
from graphlime_rdf.evaluate.agreement import (
    agreement_at_k,
    graphlime_predicate_ranking,
    predicate_of_feature,
    top_k_predicates,
)
from graphlime_rdf.evaluate.fidelity import fidelity_at_k, masked_probability
from graphlime_rdf.evaluate.stability import jaccard, jaccard_at_k, top_k_set
from graphlime_rdf.explain.baseline import gnnexplainer_predicate_ranking
from graphlime_rdf.explain.graphlime import Explanation
from graphlime_rdf.features.vocabulary import Vocabulary
from graphlime_rdf.training import TrainResult, train_run


@pytest.fixture(scope="module")
def tiny_result() -> tuple[TrainResult, RDFGraph]:
    graph = tiny_overfit_graph()
    cfg = ExperimentConfig(
        dataset="synthetic",
        feature_space=FeatureSpaceConfig(kind="predicate", directions=["out", "in"], min_support=1),
        model=RGCNModelConfig(hidden_dim=8, num_bases=4),
        training=TrainingConfig(epochs=100, weight_decay=0.0, seeds=[0]),
    )
    return train_run(graph, cfg, seed=0), graph


def test_jaccard_properties() -> None:
    assert jaccard(set(), set()) == 1.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def _explanation(names: list[str], betas: list[float]) -> Explanation:
    vocab = Vocabulary(names=tuple(names))
    return Explanation(
        node_id=0,
        beta=np.array(betas, dtype=np.float64),
        neighborhood_size=10,
        vocabulary=vocab,
    )


def test_top_k_set_and_jaccard_at_k() -> None:
    e1 = _explanation(["out:p:a", "out:p:b", "out:p:c"], [0.5, 0.3, 0.0])
    e2 = _explanation(["out:p:a", "out:p:b", "out:p:c"], [0.0, 0.4, 0.2])
    assert top_k_set(e1, 2) == {"out:p:a", "out:p:b"}
    assert top_k_set(e2, 2) == {"out:p:b", "out:p:c"}
    assert jaccard_at_k(e1, e2, 2) == pytest.approx(1 / 3)


def test_predicate_of_feature_parsing() -> None:
    assert predicate_of_feature("out:http://x.org/p#q") == "http://x.org/p#q"
    assert predicate_of_feature("in:http://x.org/p#q=<http://y.org/o>") == "http://x.org/p#q"


def test_graphlime_ranking_aggregates_directions_and_objects() -> None:
    e = _explanation(
        ["in:p:r=o1", "out:p:r=o2", "out:p:s"],
        [0.2, 0.3, 0.4],
    )
    ranking = graphlime_predicate_ranking(e)
    assert ranking[0] == ("p:r", pytest.approx(0.5))
    assert ranking[1] == ("p:s", pytest.approx(0.4))
    assert agreement_at_k(ranking, ranking, 2) == 1.0
    assert top_k_predicates(ranking, 1) == {"p:r"}


def test_masked_probability_masks_columns(tiny_result: tuple[TrainResult, RDFGraph]) -> None:
    result, graph = tiny_result
    node = graph.node_index["ex:n0"]
    probs = result.model.predict_proba(result.x, result.edge_index, result.edge_type)
    cls = int(probs[node].argmax())
    base = float(probs[node, cls])
    all_cols = list(range(result.x.shape[1]))
    # Masking every column zeroes the input entirely — prediction changes.
    p_none = masked_probability(
        result.model, result.x, result.edge_index, result.edge_type,
        node, cls, all_cols, keep_only=False,
    )
    # Keeping every column changes nothing.
    p_all = masked_probability(
        result.model, result.x, result.edge_index, result.edge_type,
        node, cls, all_cols, keep_only=True,
    )
    assert p_all == pytest.approx(base, abs=1e-6)
    assert p_none != pytest.approx(base, abs=1e-6)


def test_fidelity_deterministic(tiny_result: tuple[TrainResult, RDFGraph]) -> None:
    result, graph = tiny_result
    node = graph.node_index["ex:n0"]
    ranked = list(range(result.x.shape[1]))
    f1 = fidelity_at_k(
        result.model, result.x, result.edge_index, result.edge_type, node, ranked, k=2
    )
    f2 = fidelity_at_k(
        result.model, result.x, result.edge_index, result.edge_type, node, ranked, k=2
    )
    assert f1 == f2


def test_gnnexplainer_baseline_forwards_edge_type(
    tiny_result: tuple[TrainResult, RDFGraph],
) -> None:
    """The M0-flagged risk: edge_type must reach RGCNConv through Explainer."""
    result, graph = tiny_result
    node = graph.node_index["ex:n0"]
    ranking = gnnexplainer_predicate_ranking(
        result.model, result.x, result.edge_index, result.edge_type,
        result.vocabulary, node, epochs=20, seed=0,
    )
    assert ranking, "empty ranking"
    names = [name for name, _ in ranking]
    assert set(names) == set(graph.relation_names)  # one entry per predicate
    assert all(score >= 0 for _, score in ranking)
    # deterministic under the same seed
    again = gnnexplainer_predicate_ranking(
        result.model, result.x, result.edge_index, result.edge_type,
        result.vocabulary, node, epochs=20, seed=0,
    )
    assert ranking == again


@pytest.mark.slow
def test_fidelity_topk_beats_random_on_synthetic() -> None:
    """M7 DoD assertion — on the synthetic graph, top-k must beat random-k."""
    from pathlib import Path

    from graphlime_rdf.explain.graphlime import explain_node

    repo = Path(__file__).resolve().parents[1]
    graph = generate_ground_truth_graph(SyntheticConfig())
    cfg = ExperimentConfig.from_yaml(repo / "configs" / "synthetic.yaml")
    result = train_run(graph, cfg, seed=0)

    test_nodes = graph.test_mask.nonzero().flatten().tolist()[:30]
    gaps = []
    for node in test_nodes:
        out = explain_node(
            result.model, int(node), result.x, result.edge_index, result.edge_type,
            result.vocabulary, cfg.graphlime,
        )
        if not isinstance(out, Explanation):
            continue
        ranked = [int(j) for j in np.argsort(-out.beta, kind="stable")]
        fid_plus, _, rand_plus, _ = fidelity_at_k(
            result.model, result.x, result.edge_index, result.edge_type,
            int(node), ranked, k=2,
        )
        gaps.append(fid_plus - rand_plus)
    assert len(gaps) >= 20
    assert float(np.mean(gaps)) > 0.05, f"mean gap {float(np.mean(gaps)):.4f}"
