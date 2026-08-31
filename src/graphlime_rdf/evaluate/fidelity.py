"""Fidelity± via column masking on the interpretable features (plan M7, ADR-003).

fidelity+ = p(ŷ | full x) − p(ŷ | top-k feature columns zeroed): high means the
explanation found features the prediction *needs*. fidelity− = p(ŷ | full x) −
p(ŷ | only top-k columns kept): low means the top-k alone *suffice*. Columns
are masked graph-wide — the explanation talks about predicate signals in the
whole neighbourhood, not just the target's own row. Random-k controls use the
same masking with k columns drawn from the node-seeded RNG.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from graphlime_rdf.models.rgcn import RGCN


def _prob_of_class(
    model: RGCN, x: Tensor, edge_index: Tensor, edge_type: Tensor, node: int, cls: int
) -> float:
    return float(model.predict_proba(x, edge_index, edge_type)[node, cls])


def masked_probability(
    model: RGCN,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    node: int,
    cls: int,
    columns: list[int],
    keep_only: bool,
) -> float:
    """p(cls at node) after zeroing ``columns`` (or all *other* columns)."""
    masked = x.clone()
    if keep_only:
        keep = torch.zeros(x.shape[1], dtype=torch.bool)
        keep[columns] = True
        masked[:, ~keep] = 0.0
    else:
        masked[:, columns] = 0.0
    return _prob_of_class(model, masked, edge_index, edge_type, node, cls)


def fidelity_at_k(
    model: RGCN,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    node: int,
    ranked_columns: list[int],
    k: int,
) -> tuple[float, float, float, float]:
    """Return (fid+, fid−, random fid+, random fid−) at top-k for one node."""
    probs = model.predict_proba(x, edge_index, edge_type)
    cls = int(probs[node].argmax())
    base = float(probs[node, cls])

    top = ranked_columns[:k]
    rng = np.random.default_rng(node)
    rand = rng.choice(x.shape[1], size=min(k, x.shape[1]), replace=False).tolist()

    fid_plus = base - masked_probability(
        model, x, edge_index, edge_type, node, cls, top, keep_only=False
    )
    fid_minus = base - masked_probability(
        model, x, edge_index, edge_type, node, cls, top, keep_only=True
    )
    rand_plus = base - masked_probability(
        model, x, edge_index, edge_type, node, cls, rand, keep_only=False
    )
    rand_minus = base - masked_probability(
        model, x, edge_index, edge_type, node, cls, rand, keep_only=True
    )
    return fid_plus, fid_minus, rand_plus, rand_minus
