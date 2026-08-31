"""Interpretable feature spaces built from graph predicates (plan M2)."""

from __future__ import annotations

from torch import Tensor

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import RDFGraph
from graphlime_rdf.features.predicate import build_predicate_features
from graphlime_rdf.features.predicate_object import build_predicate_object_features
from graphlime_rdf.features.vocabulary import FeatureBuilder, Vocabulary

_BUILDERS: dict[str, FeatureBuilder] = {
    "predicate": build_predicate_features,
    "predicate_object": build_predicate_object_features,
}


def build_features(graph: RDFGraph, config: FeatureSpaceConfig) -> tuple[Tensor, Vocabulary]:
    """Dispatch to the configured feature space builder."""
    return _BUILDERS[config.kind](graph, config)


__all__ = [
    "FeatureBuilder",
    "Vocabulary",
    "build_features",
    "build_predicate_features",
    "build_predicate_object_features",
]
