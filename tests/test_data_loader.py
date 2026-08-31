"""M1 DoD tests: leakage absence, benchmark counts, determinism."""

from __future__ import annotations

import pytest
import torch

from graphlime_rdf.data.leakage import LEAKAGE_BLOCKLIST, LeakageError, assert_no_leakage
from graphlime_rdf.data.loader import RDFGraph, load_rdf_graph


@pytest.mark.parametrize("name", ["aifb", "mutag"])
def test_leakage_predicates_absent(name: str, request: pytest.FixtureRequest) -> None:
    graph: RDFGraph = request.getfixturevalue(name)
    blocked = LEAKAGE_BLOCKLIST[name]
    # Not in the name table…
    assert not blocked.intersection(graph.relation_names)
    # …and therefore unreachable from edge_type: every edge type indexes the
    # (clean) name table.
    assert int(graph.edge_type.max()) < len(graph.relation_names)
    assert int(graph.edge_type.min()) >= 0


def test_assert_no_leakage_raises_on_blocked_predicate() -> None:
    with pytest.raises(LeakageError, match="affiliation"):
        assert_no_leakage(
            "aifb", ["http://swrc.ontoware.org/ontology#affiliation"]
        )


def test_aifb_matches_published_benchmark(aifb: RDFGraph) -> None:
    assert aifb.num_nodes == 8285
    assert aifb.num_classes == 4
    assert aifb.edge_index.shape[1] == 29043  # forward triples
    assert aifb.num_relations == 45
    assert int(aifb.train_mask.sum()) == 140
    assert int(aifb.test_mask.sum()) == 36


def test_mutag_matches_published_benchmark(mutag: RDFGraph) -> None:
    assert mutag.num_nodes == 23644
    assert mutag.num_classes == 2
    assert mutag.edge_index.shape[1] == 74227
    assert mutag.num_relations == 23
    assert int(mutag.train_mask.sum()) == 272
    assert int(mutag.test_mask.sum()) == 68


def test_masks_disjoint_and_labels_consistent(aifb: RDFGraph) -> None:
    assert not bool((aifb.train_mask & aifb.test_mask).any())
    labeled = aifb.train_mask | aifb.test_mask
    assert bool((aifb.labels[labeled] >= 0).all())
    assert bool((aifb.labels[~labeled] == -1).all())


def test_loader_deterministic_across_calls(aifb: RDFGraph) -> None:
    again = load_rdf_graph("aifb")
    assert torch.equal(aifb.edge_index, again.edge_index)
    assert torch.equal(aifb.edge_type, again.edge_type)
    assert torch.equal(aifb.labels, again.labels)
    assert aifb.relation_names == again.relation_names
    assert aifb.node_names == again.node_names
    assert aifb.label_names == again.label_names


def test_relation_names_are_readable_uris(aifb: RDFGraph) -> None:
    assert all(name.startswith("http") for name in aifb.relation_names)


def test_doubled_edges(aifb: RDFGraph) -> None:
    edge_index, edge_type = aifb.doubled_edges()
    e = aifb.edge_index.shape[1]
    assert edge_index.shape == (2, 2 * e)
    assert edge_type.shape == (2 * e,)
    # Inverse block is the flipped forward block with odd relation ids.
    assert torch.equal(edge_index[:, :e], aifb.edge_index)
    assert torch.equal(edge_index[0, e:], aifb.edge_index[1])
    assert torch.equal(edge_index[1, e:], aifb.edge_index[0])
    assert torch.equal(edge_type[:e] % 2, torch.zeros(e, dtype=torch.long))
    assert torch.equal(edge_type[e:] % 2, torch.ones(e, dtype=torch.long))


def test_unknown_dataset_rejected() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        load_rdf_graph("nope")
