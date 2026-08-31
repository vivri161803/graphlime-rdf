"""Smoke test: the package imports and declares its purpose."""

import graphlime_rdf


def test_package_imports() -> None:
    assert graphlime_rdf.__doc__ is not None
    assert "GraphLIME" in graphlime_rdf.__doc__
