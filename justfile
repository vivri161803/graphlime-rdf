# Task runner for graphlime-rdf. All quality gates live here.

set shell := ["bash", "-cu"]

fmt:
    uv run ruff format src tests

lint:
    uv run ruff check --fix src tests

types:
    uv run mypy

test:
    uv run pytest -m "not slow" --cov=src --cov-fail-under=85

slow:
    uv run pytest -m slow

check: lint types test

# M6 synthetic ground-truth gate: GraphLIME must recover the planted predicate.
synthetic:
    uv run pytest tests/test_synthetic_gate.py -m slow -x

# Full deterministic reproduction: AIFB + MUTAG end-to-end, regenerates results/.
repro:
    uv run python -m graphlime_rdf.cli repro

tables:
    uv run python -m graphlime_rdf.cli tables

readme:
    uv run python -m graphlime_rdf.cli readme

render:
    uv run jupytext --to ipynb --execute notebooks/qualitative.py -o notebooks/qualitative.ipynb
    uv run jupyter nbconvert --to html notebooks/qualitative.ipynb --output qualitative.html --output-dir notebooks
