"""Explanation stability: Jaccard@k of top-k feature sets (plan M7).

Compared across training seeds (same node, models from different seeds) and
across neighbourhood radii (hops ∈ {1,2,3}).
"""

from __future__ import annotations

from graphlime_rdf.explain.graphlime import Explanation


def top_k_set(explanation: Explanation, k: int) -> set[str]:
    return {name for name, _ in explanation.top_features(k)}


def jaccard(a: set[str], b: set[str]) -> float:
    """|a ∩ b| / |a ∪ b|; two empty sets count as perfectly agreeing."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def jaccard_at_k(e1: Explanation, e2: Explanation, k: int) -> float:
    return jaccard(top_k_set(e1, k), top_k_set(e2, k))
