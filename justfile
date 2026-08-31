# Task runner for graphlime-rdf. All quality gates live here.

set shell := ["bash", "-cu"]

# Auto-format the code with ruff.
fmt:
    uv run ruff format src tests

# Lint and auto-fix with ruff.
lint:
    uv run ruff check --fix src tests

# Static type check (mypy --strict).
types:
    uv run mypy

# Fast tests with the 85% coverage gate.
test:
    uv run pytest -m "not slow" --cov=src --cov-fail-under=85

# The slow tests only (training, synthetic gate).
slow:
    uv run pytest -m slow

# The pre-commit gate: lint + types + test. Must be green before every commit.
check: lint types test

# M6 synthetic ground-truth gate: GraphLIME must recover the planted predicate.
synthetic:
    uv run pytest tests/test_synthetic_gate.py -m slow -x

# Full deterministic reproduction: AIFB + MUTAG end-to-end, regenerates results/.
repro:
    uv run python -m graphlime_rdf.cli repro

# Regenerate results/tables/ and results/figures/ from the JSONL records.
tables:
    uv run python -m graphlime_rdf.cli tables

# Inject the current tables into report/relazione.md.
readme:
    uv run python -m graphlime_rdf.cli readme

# PDF of the report (pandoc + xelatex); run `just readme` first if results/ changed.
pdf:
    cd report && pandoc relazione.md -o relazione.pdf \
        --pdf-engine=xelatex \
        -V mainfont="Palatino" \
        -V monofont="Menlo" -V monofontoptions="Scale=0.85" \
        -V geometry:margin=2.5cm -V fontsize=11pt -V linkcolor=blue \
        -H pdf-preamble.tex

# Slide deck PDF (beamer/metropolis); runs twice so the outline resolves.
slides:
    cd report && pdflatex -interaction=nonstopmode -halt-on-error presentazione.tex
    cd report && pdflatex -interaction=nonstopmode -halt-on-error presentazione.tex

# Execute the qualitative notebook and render it to HTML.
render:
    uv run python -m ipykernel install --user --name graphlime-rdf
    uv run jupytext --to ipynb --execute --set-kernel graphlime-rdf notebooks/qualitative.py -o notebooks/qualitative.ipynb
    uv run jupyter nbconvert --to html notebooks/qualitative.ipynb --output qualitative.html --output-dir notebooks
