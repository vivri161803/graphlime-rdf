"""Feature space B: (predicate, object) pairs with support pruning (plan M2).

Feature ``out:<p>=<o>`` marks a node having predicate ``p`` toward the specific
term ``o``; ``in:<p>=<s>`` marks being the object of ``p`` from ``s``. Pairs
seen on fewer than ``min_support`` distinct nodes are pruned. Strictly more
expressive than space A (every A feature is the union of B features sharing the
predicate, when ``min_support=1``).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
from torch import Tensor

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.features.vocabulary import Vocabulary


def feature_name(direction: str, predicate: str, term: str) -> str:
    return f"{direction}:{predicate}={term}"


def build_predicate_object_features(
    graph: RDFGraph, config: FeatureSpaceConfig
) -> tuple[Tensor, Vocabulary]:
    """Return the (num_nodes, D) pair-indicator matrix and its vocabulary."""
    src, dst = graph.edge_index
    # feature name → set of nodes carrying it (support = distinct nodes)
    carriers: dict[str, set[int]] = defaultdict(set)
    counts: Counter[tuple[int, str]] = Counter()
    for e in range(graph.edge_type.shape[0]):
        p = graph.relation_names[int(graph.edge_type[e])]
        s, o = int(src[e]), int(dst[e])
        for direction in config.directions:
            if direction == "out":
                name = feature_name("out", p, graph.node_names[o])
                node = s
            else:
                name = feature_name("in", p, graph.node_names[s])
                node = o
            carriers[name].add(node)
            counts[node, name] += 1

    kept = [n for n, nodes in carriers.items() if len(nodes) >= config.min_support]
    vocab = Vocabulary.from_names(kept)
    col = {name: j for j, name in enumerate(vocab.names)}

    x = torch.zeros((graph.num_nodes, len(vocab)), dtype=torch.float32)
    for (node, name), c in counts.items():
        j = col.get(name)
        if j is not None:
            x[node, j] = 1.0 if config.binary else float(c)
    return x, vocab
