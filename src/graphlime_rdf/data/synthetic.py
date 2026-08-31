"""Synthetic graphs with known ground truth (plan M6) and fixture helpers.

``graph_from_triples`` turns a plain triple list into a typed
:class:`RDFGraph` with the same deterministic indexing as the real loader —
used by unit-test fixtures and by the ground-truth generator.
"""

from __future__ import annotations

import numpy as np
import torch

from graphlime_rdf.config import SyntheticConfig
from graphlime_rdf.data.leakage import assert_no_leakage
from graphlime_rdf.data.loader import RDFGraph


def generate_ground_truth_graph(config: SyntheticConfig) -> RDFGraph:
    """Entities with random predicates; class 1 ⟺ has the target predicate.

    Every predicate ``p<r>`` links an entity to the shared object ``obj<r>``,
    so the ground truth for an explanation of any entity is exactly the
    feature ``out:syn:p<target>``. Distractor predicates ``d<j>`` co-occur
    with the target with ``distractor_strength`` — plausible red herrings.
    """
    rng = np.random.default_rng(config.seed)
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, int] = {}
    t = config.target_predicate

    for i in range(config.num_entities):
        entity = f"syn:e{i:04d}"
        cls = int(rng.random() < config.class_balance)
        if cls == 1:
            triples.append((entity, f"syn:p{t}", f"syn:obj{t}"))
        for r in range(config.num_predicates):
            if r == t:
                continue
            if rng.random() < config.edge_prob:
                triples.append((entity, f"syn:p{r}", f"syn:obj{r}"))
        for j in range(config.num_distractors):
            p_present = (
                config.distractor_strength if cls == 1 else 1 - config.distractor_strength
            )
            if rng.random() < p_present:
                triples.append((entity, f"syn:d{j}", f"syn:dobj{j}"))
        observed = cls if rng.random() >= config.noise_rate else 1 - cls
        labels[entity] = observed

    entities = sorted(labels)
    n_train = int(config.train_fraction * len(entities))
    perm = rng.permutation(len(entities))
    train_nodes = {entities[int(k)] for k in perm[:n_train]}
    test_nodes = {entities[int(k)] for k in perm[n_train:]}
    return graph_from_triples(
        triples, labels=labels, train_nodes=train_nodes, test_nodes=test_nodes
    )


def tiny_overfit_graph() -> RDFGraph:
    """10-node graph the R-GCN must drive to 100% train accuracy (M3 DoD).

    Class 1 ⟺ the entity has predicate ``ex:special``; class 0 entities carry
    ``ex:other`` instead. Perfectly separable in the predicate feature space.
    """
    triples: list[tuple[str, str, str]] = []
    labels: dict[str, int] = {}
    for i in range(8):
        node = f"ex:n{i}"
        cls = i % 2
        pred = "ex:special" if cls == 1 else "ex:other"
        target = "ex:hub1" if cls == 1 else "ex:hub0"
        triples.append((node, pred, target))
        triples.append((node, "ex:linked", f"ex:n{(i + 1) % 8}"))
        labels[node] = cls
    train = {f"ex:n{i}" for i in range(6)}
    test = {"ex:n6", "ex:n7"}
    return graph_from_triples(triples, labels=labels, train_nodes=train, test_nodes=test)


def graph_from_triples(
    triples: list[tuple[str, str, str]],
    labels: dict[str, int] | None = None,
    train_nodes: set[str] | None = None,
    test_nodes: set[str] | None = None,
    dataset: str = "synthetic",
) -> RDFGraph:
    """Build a deterministic :class:`RDFGraph` from string triples.

    Node/relation indices follow lexicographic order exactly like the real
    loader. ``labels`` maps node name → class id; unlabeled nodes get ``-1``.
    """
    labels = labels or {}
    train_nodes = train_nodes if train_nodes is not None else set(labels)
    test_nodes = test_nodes or set()

    relation_names = sorted({p for _, p, _ in triples})
    assert_no_leakage(dataset, relation_names)
    node_names = sorted({s for s, _, _ in triples} | {o for _, _, o in triples})
    node_index = {n: i for i, n in enumerate(node_names)}
    relation_index = {r: i for i, r in enumerate(relation_names)}

    ordered = sorted(triples)
    edge_index = torch.tensor(
        [[node_index[s] for s, _, _ in ordered], [node_index[o] for _, _, o in ordered]],
        dtype=torch.long,
    )
    edge_type = torch.tensor([relation_index[p] for _, p, _ in ordered], dtype=torch.long)

    n = len(node_names)
    label_tensor = torch.full((n,), -1, dtype=torch.long)
    train_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    for name, cls in labels.items():
        label_tensor[node_index[name]] = cls
    for name in train_nodes:
        train_mask[node_index[name]] = True
    for name in test_nodes:
        test_mask[node_index[name]] = True

    num_classes = max(labels.values()) + 1 if labels else 0
    return RDFGraph(
        dataset=dataset,
        edge_index=edge_index,
        edge_type=edge_type,
        relation_names=relation_names,
        node_names=node_names,
        labels=label_tensor,
        label_names=[f"class_{c}" for c in range(num_classes)],
        train_mask=train_mask,
        test_mask=test_mask,
        node_index=node_index,
    )
