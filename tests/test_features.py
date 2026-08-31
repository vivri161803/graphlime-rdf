"""M2 DoD tests: determinism, readable names, counts, pruning, A ⊆ B."""

from __future__ import annotations

import torch

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.data.synthetic import graph_from_triples
from graphlime_rdf.features import build_features
from graphlime_rdf.features.vocabulary import Vocabulary

# Hand-built 6-triple fixture (plan M2 DoD): two people, two topics.
FIXTURE_TRIPLES = [
    ("ex:anna", "ex:worksOn", "ex:graphs"),
    ("ex:anna", "ex:knows", "ex:ben"),
    ("ex:ben", "ex:worksOn", "ex:logic"),
    ("ex:ben", "ex:knows", "ex:anna"),
    ("ex:anna", "ex:memberOf", "ex:lab1"),
    ("ex:ben", "ex:memberOf", "ex:lab1"),
]


def fixture_graph() -> RDFGraph:
    return graph_from_triples(FIXTURE_TRIPLES, labels={"ex:anna": 0, "ex:ben": 1})


def test_vocabulary_sorted_and_hash_stable() -> None:
    v1 = Vocabulary.from_names({"b", "a", "c"})
    v2 = Vocabulary.from_names(["c", "a", "b", "a"])
    assert v1.names == ("a", "b", "c")
    assert v1 == v2
    assert v1.hash == v2.hash
    assert v1.index("b") == 1


def test_vocab_identical_across_runs() -> None:
    graph = fixture_graph()
    for kind in ["predicate", "predicate_object"]:
        cfg = FeatureSpaceConfig(kind=kind, min_support=1)  # type: ignore[arg-type]
        x1, v1 = build_features(graph, cfg)
        x2, v2 = build_features(graph, cfg)
        assert v1 == v2
        assert torch.equal(x1, x2)


def test_every_feature_maps_to_readable_name(aifb: RDFGraph) -> None:
    for kind in ["predicate", "predicate_object"]:
        cfg = FeatureSpaceConfig(kind=kind)  # type: ignore[arg-type]
        x, vocab = build_features(aifb, cfg)
        assert x.shape == (aifb.num_nodes, len(vocab))
        for name in vocab.names:
            direction, rest = name.split(":", 1)
            assert direction in {"out", "in"}
            assert rest.startswith("http"), f"not a readable URI: {name}"


def test_matrix_row_count_is_entity_count() -> None:
    graph = fixture_graph()
    for kind in ["predicate", "predicate_object"]:
        cfg = FeatureSpaceConfig(kind=kind, min_support=1)  # type: ignore[arg-type]
        x, vocab = build_features(graph, cfg)
        assert x.shape == (graph.num_nodes, len(vocab))


def test_predicate_features_content() -> None:
    graph = fixture_graph()
    x, vocab = build_features(graph, FeatureSpaceConfig(kind="predicate"))
    anna = graph.node_index["ex:anna"]
    ben = graph.node_index["ex:ben"]
    assert x[anna, vocab.index("out:ex:worksOn")] == 1.0
    assert x[ben, vocab.index("out:ex:worksOn")] == 1.0
    assert x[anna, vocab.index("out:ex:memberOf")] == 1.0
    # topics have no outgoing edges at all
    graphs_node = graph.node_index["ex:graphs"]
    assert bool((x[graphs_node] == 0).all())


def test_in_direction_features() -> None:
    graph = fixture_graph()
    cfg = FeatureSpaceConfig(kind="predicate", directions=["out", "in"])
    x, vocab = build_features(graph, cfg)
    graphs_node = graph.node_index["ex:graphs"]
    assert x[graphs_node, vocab.index("in:ex:worksOn")] == 1.0
    assert x[graphs_node, vocab.index("in:ex:knows")] == 0.0


def test_pruning_monotone_in_min_support() -> None:
    graph = fixture_graph()
    sizes = []
    for ms in [1, 2, 3]:
        _, vocab = build_features(
            graph, FeatureSpaceConfig(kind="predicate_object", min_support=ms)
        )
        sizes.append(len(vocab))
    assert sizes == sorted(sizes, reverse=True)
    # memberOf=lab1 is carried by 2 nodes: survives ms=2, dies at ms=3.
    _, v2 = build_features(graph, FeatureSpaceConfig(kind="predicate_object", min_support=2))
    assert v2.names == ("out:ex:memberOf=ex:lab1",)


def test_space_b_at_least_as_expressive_as_space_a() -> None:
    """Any pair of nodes distinguished by A is distinguished by B (ms=1)."""
    graph = fixture_graph()
    xa, _ = build_features(graph, FeatureSpaceConfig(kind="predicate"))
    xb, _ = build_features(graph, FeatureSpaceConfig(kind="predicate_object", min_support=1))
    for i in range(graph.num_nodes):
        for j in range(graph.num_nodes):
            if not torch.equal(xa[i], xa[j]):
                assert not torch.equal(xb[i], xb[j]), (i, j)


def test_binary_false_counts_multiplicity() -> None:
    triples = [*FIXTURE_TRIPLES, ("ex:anna", "ex:worksOn", "ex:logic")]
    graph = graph_from_triples(triples)
    x, vocab = build_features(graph, FeatureSpaceConfig(kind="predicate", binary=False))
    anna = graph.node_index["ex:anna"]
    assert x[anna, vocab.index("out:ex:worksOn")] == 2.0
