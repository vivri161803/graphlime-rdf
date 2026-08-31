"""M5 DoD tests: neighbourhood extraction, refusals, output shape, determinism."""

from __future__ import annotations

import numpy as np
import torch

from graphlime_rdf.config import (
    ExperimentConfig,
    FeatureSpaceConfig,
    GraphLIMEConfig,
    RGCNModelConfig,
    TrainingConfig,
)
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.data.synthetic import graph_from_triples
from graphlime_rdf.explain.graphlime import Explanation, Refusal, explain_node, neighborhood
from graphlime_rdf.training import TrainResult, train_run

# A path graph a - b - c - d plus a star around b, so hop counts are easy to
# compute by hand.
PATH_TRIPLES = [
    ("ex:a", "ex:r", "ex:b"),
    ("ex:b", "ex:r", "ex:c"),
    ("ex:c", "ex:r", "ex:d"),
    ("ex:b", "ex:s", "ex:s1"),
    ("ex:b", "ex:s", "ex:s2"),
]


def _path_graph_result(epochs: int = 5) -> tuple[TrainResult, RDFGraph]:
    graph = graph_from_triples(
        PATH_TRIPLES, labels={"ex:a": 0, "ex:b": 1, "ex:c": 0, "ex:d": 1}
    )
    cfg = ExperimentConfig(
        dataset="synthetic",
        feature_space=FeatureSpaceConfig(kind="predicate", directions=["out", "in"], min_support=1),
        model=RGCNModelConfig(hidden_dim=4, num_bases=2),
        training=TrainingConfig(epochs=epochs, seeds=[0]),
    )
    return train_run(graph, cfg, seed=0), graph


def test_neighborhood_matches_hand_computed_values() -> None:
    result, graph = _path_graph_result()
    a = graph.node_index["ex:a"]

    one_hop = neighborhood(a, 1, result.edge_index, graph.num_nodes)
    assert {graph.node_names[int(i)] for i in one_hop} == {"ex:a", "ex:b"}

    two_hop = neighborhood(a, 2, result.edge_index, graph.num_nodes)
    assert {graph.node_names[int(i)] for i in two_hop} == {
        "ex:a", "ex:b", "ex:c", "ex:s1", "ex:s2",
    }


def test_target_always_included() -> None:
    result, graph = _path_graph_result()
    for name in ["ex:a", "ex:b", "ex:d"]:
        node = graph.node_index[name]
        for hops in [1, 2, 3]:
            nodes = neighborhood(node, hops, result.edge_index, graph.num_nodes)
            assert node in nodes.tolist()


def test_refusal_below_min_neighborhood_with_reason() -> None:
    result, graph = _path_graph_result()
    a = graph.node_index["ex:a"]
    cfg = GraphLIMEConfig(hops=1, min_neighborhood=10)
    out = explain_node(
        result.model, a, result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg,
    )
    assert isinstance(out, Refusal)
    assert out.neighborhood_size == 2
    assert "min_neighborhood" in out.reason


def test_explanation_shape_and_determinism() -> None:
    result, graph = _path_graph_result()
    b = graph.node_index["ex:b"]
    cfg = GraphLIMEConfig(hops=2, min_neighborhood=2)
    out1 = explain_node(
        result.model, b, result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg,
    )
    out2 = explain_node(
        result.model, b, result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg,
    )
    assert isinstance(out1, Explanation) and isinstance(out2, Explanation)
    assert out1.beta.shape == (len(result.vocabulary),)
    np.testing.assert_array_equal(out1.beta, out2.beta)
    assert np.all(out1.beta >= 0)
    assert 0.0 <= out1.sparsity <= 1.0
    for name, weight in out1.top_features(3):
        assert isinstance(name, str) and ":" in name
        assert weight > 0


def test_subsampling_is_deterministic_and_keeps_target() -> None:
    # Star graph: hub connected to 60 leaves; cap at 20.
    triples = [(f"ex:leaf{i}", "ex:r", "ex:hub") for i in range(60)]
    graph = graph_from_triples(triples, labels={"ex:hub": 0, "ex:leaf0": 1})
    cfg_exp = ExperimentConfig(
        dataset="synthetic",
        feature_space=FeatureSpaceConfig(kind="predicate", directions=["out", "in"], min_support=1),
        model=RGCNModelConfig(hidden_dim=4, num_bases=2),
        training=TrainingConfig(epochs=3, seeds=[0]),
    )
    result = train_run(graph, cfg_exp, seed=0)
    hub = graph.node_index["ex:hub"]
    cfg = GraphLIMEConfig(hops=1, min_neighborhood=2, max_neighborhood=20)
    out1 = explain_node(
        result.model, hub, result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg,
    )
    out2 = explain_node(
        result.model, hub, result.x, result.edge_index, result.edge_type,
        result.vocabulary, cfg,
    )
    assert isinstance(out1, Explanation) and isinstance(out2, Explanation)
    assert out1.neighborhood_size == 61  # reported size is pre-cap
    np.testing.assert_array_equal(out1.beta, out2.beta)


def test_interpretable_x_dimension_checked() -> None:
    import pytest

    result, graph = _path_graph_result()
    b = graph.node_index["ex:b"]
    bad_x = torch.zeros((graph.num_nodes, 3))
    with pytest.raises(ValueError, match="vocabulary"):
        explain_node(
            result.model, b, result.x, result.edge_index, result.edge_type,
            result.vocabulary, GraphLIMEConfig(), interpretable_x=bad_x,
        )
