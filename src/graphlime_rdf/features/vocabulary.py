"""Deterministic feature vocabulary and the FeatureBuilder protocol (plan M2).

A vocabulary is an ordered list of **human-readable** feature names; feature
``j`` of the matrix is always ``vocabulary.names[j]``. Names are sorted, so the
index assignment is independent of insertion order and ``PYTHONHASHSEED``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from graphlime_rdf.config import FeatureSpaceConfig
from graphlime_rdf.data.loader import RDFGraph


@dataclass(frozen=True)
class Vocabulary:
    """Ordered, hash-stable mapping between feature indices and names."""

    names: tuple[str, ...]

    @staticmethod
    def from_names(names: set[str] | list[str]) -> Vocabulary:
        return Vocabulary(names=tuple(sorted(set(names))))

    def __len__(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        return self.names.index(name)

    @property
    def hash(self) -> str:
        """Stable digest — embedded in checkpoint manifests."""
        h = hashlib.sha256("\n".join(self.names).encode())
        return h.hexdigest()[:16]


class FeatureBuilder(Protocol):
    """Space A and space B implement this one protocol."""

    def __call__(
        self, graph: RDFGraph, config: FeatureSpaceConfig
    ) -> tuple[Tensor, Vocabulary]:
        """Return ``(X, vocabulary)`` with ``X.shape == (num_nodes, len(vocabulary))``."""
        ...
