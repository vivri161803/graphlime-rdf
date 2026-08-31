"""GraphLIME: local explanations via HSIC Lasso on the k-hop neighbourhood (plan M5).

The samples are the nodes of the target's k-hop subgraph (target included —
its own features/prediction pair carries signal too, following the original
GraphLIME formulation). ``X`` holds their interpretable features, ``Y`` the
model's predicted probability vectors; the non-negative HSIC Lasso weights
say which *named predicates* the prediction depends on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import k_hop_subgraph

from graphlime_rdf.config import GraphLIMEConfig
from graphlime_rdf.explain.hsic import FloatArray
from graphlime_rdf.explain.hsic_lasso import hsic_lasso
from graphlime_rdf.features.vocabulary import Vocabulary
from graphlime_rdf.models.rgcn import RGCN


@dataclass(frozen=True)
class Explanation:
    """A successful GraphLIME explanation for one node."""

    node_id: int
    beta: FloatArray  # (len(vocabulary),) — β ≥ 0, aligned with vocab indices
    neighborhood_size: int
    vocabulary: Vocabulary

    def top_features(self, k: int) -> list[tuple[str, float]]:
        """The k highest-weight features as (human-readable name, β) pairs."""
        order = np.argsort(-self.beta, kind="stable")[:k]
        return [(self.vocabulary.names[j], float(self.beta[j])) for j in order if self.beta[j] > 0]

    @property
    def sparsity(self) -> float:
        """Fraction of features with zero weight (higher = sparser)."""
        return float(np.mean(self.beta <= 1e-12))


@dataclass(frozen=True)
class Refusal:
    """A refused explanation — counted and reported, never silently dropped."""

    node_id: int
    reason: str
    neighborhood_size: int


def neighborhood(node: int, hops: int, edge_index: Tensor, num_nodes: int) -> Tensor:
    """Node ids of the k-hop subgraph around ``node`` (target included, sorted)."""
    subset, _, _, _ = k_hop_subgraph(
        node, hops, edge_index, relabel_nodes=False, num_nodes=num_nodes
    )
    return torch.sort(subset)[0]


def explain_node(
    model: RGCN,
    node: int,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    vocabulary: Vocabulary,
    config: GraphLIMEConfig,
    interpretable_x: Tensor | None = None,
) -> Explanation | Refusal:
    """Explain the model's prediction at ``node`` in the vocabulary's terms.

    ``x`` is the model input; ``interpretable_x`` (defaults to ``x``) is the
    feature matrix GraphLIME regresses on — passing a different matrix is the
    feature-space sensitivity experiment (A vs B, plan M7).
    """
    if interpretable_x is None:
        interpretable_x = x
    if interpretable_x.shape[1] != len(vocabulary):
        raise ValueError(
            f"interpretable feature dim {interpretable_x.shape[1]} != vocabulary {len(vocabulary)}"
        )

    nodes = neighborhood(node, config.hops, edge_index, x.shape[0])
    size = int(nodes.shape[0])
    if size < config.min_neighborhood:
        return Refusal(
            node_id=node,
            reason=f"neighborhood has {size} nodes < min_neighborhood={config.min_neighborhood}",
            neighborhood_size=size,
        )
    if size > config.max_neighborhood:
        nodes = _subsample(nodes, node, config.max_neighborhood)

    probs = model.predict_proba(x, edge_index, edge_type)
    samples_x = interpretable_x[nodes].numpy().astype(np.float64)
    samples_y = probs[nodes].numpy().astype(np.float64)

    # Columns constant across the samples provably get β = 0 (their centered
    # kernel is the zero matrix) — solve only the active ones, scatter back.
    active = np.flatnonzero(samples_x.std(axis=0) > 0)
    beta = np.zeros(len(vocabulary), dtype=np.float64)
    if active.size:
        beta[active] = hsic_lasso(
            samples_x[:, active], samples_y, rho=config.rho, sigma_x=config.sigma
        )
    return Explanation(
        node_id=node, beta=beta, neighborhood_size=size, vocabulary=vocabulary
    )


def _subsample(nodes: Tensor, target: int, cap: int) -> Tensor:
    """Deterministic subsample (seeded by target id), target always kept."""
    others = nodes[nodes != target].numpy()
    rng = np.random.default_rng(target)
    chosen = rng.choice(others, size=cap - 1, replace=False)
    return torch.sort(torch.tensor(np.append(chosen, target), dtype=torch.long))[0]
