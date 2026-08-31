"""Explicit leakage blocklist per dataset (plan M1, ADR-002).

The AIFB/MUTAG benchmark dumps published for the R-GCN paper (Schlichtkrull
et al., 2018) are the ``*_stripped`` variants: the predicates that *are* the
classification target were removed by the benchmark authors, because a model
that sees them reads the label off the graph instead of learning it.

We assert their absence at load time rather than silently trusting the files:
if upstream ever swaps in unstripped dumps, loading fails loudly.
"""

from __future__ import annotations

# Why each predicate is blocked:
# - AIFB: the task is predicting a person's research-group affiliation.
#   `swrc:affiliation` states it directly; `swrc:employs` is its inverse
#   (group → person), equally label-revealing.
# - MUTAG: the task is predicting mutagenicity of a compound.
#   `carcinogenesis#isMutagenic` states the label itself.
LEAKAGE_BLOCKLIST: dict[str, frozenset[str]] = {
    "aifb": frozenset(
        {
            "http://swrc.ontoware.org/ontology#affiliation",
            "http://swrc.ontoware.org/ontology#employs",
        }
    ),
    "mutag": frozenset(
        {
            "http://dl-learner.org/carcinogenesis#isMutagenic",
        }
    ),
    "synthetic": frozenset(),
}


class LeakageError(RuntimeError):
    """Raised when a label-revealing predicate is found in the loaded graph."""


def assert_no_leakage(dataset: str, relation_names: list[str]) -> None:
    """Fail loudly if any blocked predicate appears among the relations."""
    blocked = LEAKAGE_BLOCKLIST[dataset]
    present = blocked.intersection(relation_names)
    if present:
        raise LeakageError(
            f"Leakage predicates present in {dataset!r}: {sorted(present)}. "
            "The loaded dump is not the stripped benchmark file."
        )
