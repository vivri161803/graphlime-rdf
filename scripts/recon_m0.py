"""M0 API reconnaissance: verify PyG APIs against the installed version.

Answers the six questions in the implementation plan §4 and prints findings
that are transcribed into docs/api-notes.md.
"""

from __future__ import annotations

import importlib.metadata as md
import inspect

import torch
import torch_geometric


def main() -> None:
    print("=== versions ===")
    for pkg in [
        "torch",
        "torch-geometric",
        "numpy",
        "scipy",
        "scikit-learn",
        "pandas",
        "pydantic",
        "rdflib",
        "typer",
        "jupytext",
        "nbconvert",
    ]:
        try:
            print(f"{pkg}=={md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"{pkg}: NOT INSTALLED")

    print("\n=== Q3: explainers shipped ===")
    import torch_geometric.explain.algorithm as alg

    print([n for n in dir(alg) if not n.startswith("_") and n[0].isupper()])

    print("\n=== Q4: fidelity metric helpers ===")
    import torch_geometric.explain.metric as met

    print([n for n in dir(met) if not n.startswith("_")])
    try:
        from torch_geometric.explain.metric import fidelity

        print("fidelity signature:", inspect.signature(fidelity))
        print(inspect.getsource(fidelity)[:1500])
    except ImportError as e:
        print("no fidelity helper:", e)

    print("\n=== Q1/Q2: Entities dataset source ===")
    from torch_geometric.datasets import Entities

    src = inspect.getsource(Entities)
    print(src)

    print("\n=== torch determinism check ===")
    torch.use_deterministic_algorithms(True)
    print("use_deterministic_algorithms OK")


if __name__ == "__main__":
    main()
