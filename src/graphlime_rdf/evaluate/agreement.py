"""Agreement metrics: feature-space sensitivity and explainer agreement (plan M7).

Both comparisons happen at the level of **predicate URIs** — GraphLIME feature
names are collapsed by stripping the direction prefix and the ``=object``
suffix, GNNExplainer edge masks are aggregated per relation — so top-k sets
from any source are directly comparable.
"""

from __future__ import annotations

from collections import defaultdict

from graphlime_rdf.evaluate.stability import jaccard
from graphlime_rdf.explain.graphlime import Explanation


def predicate_of_feature(name: str) -> str:
    """'out:<p>=<o>' or 'in:<p>' → '<p>'."""
    _, rest = name.split(":", 1)
    return rest.split("=", 1)[0]


def graphlime_predicate_ranking(explanation: Explanation) -> list[tuple[str, float]]:
    """Aggregate β by predicate (summing directions / objects), ranked."""
    totals: dict[str, float] = defaultdict(float)
    for name, beta in explanation.top_features(len(explanation.beta)):
        totals[predicate_of_feature(name)] += beta
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))


def top_k_predicates(ranking: list[tuple[str, float]], k: int) -> set[str]:
    return {name for name, _ in ranking[:k]}


def agreement_at_k(
    ranking_a: list[tuple[str, float]], ranking_b: list[tuple[str, float]], k: int
) -> float:
    """Jaccard of the two top-k predicate sets."""
    return jaccard(top_k_predicates(ranking_a, k), top_k_predicates(ranking_b, k))
