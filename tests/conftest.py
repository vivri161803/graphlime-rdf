"""Shared fixtures: real benchmark graphs are loaded once per session."""

from __future__ import annotations

import pytest

from graphlime_rdf.data.loader import RDFGraph, load_rdf_graph


@pytest.fixture(scope="session")
def aifb() -> RDFGraph:
    return load_rdf_graph("aifb")


@pytest.fixture(scope="session")
def mutag() -> RDFGraph:
    return load_rdf_graph("mutag")
