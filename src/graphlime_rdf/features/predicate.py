"""Feature space A: predicate indicators (plan M2).

Feature ``out:<p>`` marks that a node has at least one outgoing edge with
predicate ``p`` (count if ``binary=False``); ``in:<p>`` likewise for incoming
edges. This is the interpretable input the R-GCN consumes (plan §6).
"""

from __future__ import annotations

import torch
from torch import Tensor

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.features.vocabulary import Vocabulary


def feature_name(direction: str, predicate: str) -> str:
    return f"{direction}:{predicate}"


def build_predicate_features(
    graph: RDFGraph, config: FeatureSpaceConfig
) -> tuple[Tensor, Vocabulary]:
    """Return the (num_nodes, D) predicate-indicator matrix and its vocabulary."""
    names: set[str] = set()
    for direction in config.directions:
        for p in graph.relation_names:
            names.add(feature_name(direction, p))
    vocab = Vocabulary.from_names(names)
    col = {name: j for j, name in enumerate(vocab.names)}

    x = torch.zeros((graph.num_nodes, len(vocab)), dtype=torch.float32)
    src, dst = graph.edge_index
    for direction in config.directions:
        node_of_edge = src if direction == "out" else dst
        for e in range(graph.edge_type.shape[0]):
            p = graph.relation_names[int(graph.edge_type[e])]
            j = col[feature_name(direction, p)]
            x[int(node_of_edge[e]), j] += 1.0
    if config.binary:
        x = (x > 0).to(torch.float32)
    return x, vocab
