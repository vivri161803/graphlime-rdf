# graphlime-rdf

**GraphLIME explanations for R-GCN entity classification on RDF knowledge
graphs — expressed in the vocabulary of the graph itself.**

An R-GCN normally classifies RDF entities from one-hot ids: accurate, but
unexplainable in principle — no attribution method can make "entity #4211"
meaningful. This project replaces the input with an **explicit interpretable
feature space built from graph predicates** (counts of `swrc:publication`,
`carcinogenesis:hasAtom`, …), trains the R-GCN on it, and explains individual
predictions with a from-scratch **GraphLIME** (HSIC Lasso). Every explanation
is a ranked list of *named predicates*; an integer index where a name should
be is, by project rule, a bug.

Everything below is verified: a synthetic ground-truth gate had to pass
before any real-data number was produced, property tests (written before the
implementation) pin down the HSIC math, and every reported number is
generated from validated JSONL records carrying the config hash, seed, and
git commit that produced it.

## Results at a glance

<!-- RESULTS:BEGIN -->
<!-- RESULTS:END -->

## Method in one paragraph

Two interpretable feature spaces are built from the triples: **A** —
per-predicate edge counts (`out:<p>`, `in:<p>`), and **B** —
predicate–object pairs (`out:<p>=<o>`, support-pruned). The R-GCN
(2-layer `RGCNConv`, basis decomposition) consumes space A as its node
features, so model and explainer share one vocabulary. To explain a node,
GraphLIME treats its k-hop neighbours as local samples and selects, with a
non-negative HSIC Lasso (kernels, centering, and solver implemented from
scratch — `docs/method.md`), the features whose similarity structure best
explains the model's output similarity structure. Fidelity is measured by
masking feature columns of the actual model input, against random-k
controls; GNNExplainer (attribute mask) is the baseline. Correctness is
gated on a synthetic graph where the ground-truth predicate is known
(`just synthetic`).

## Repository map

```
configs/          experiment YAMLs (single source of hyperparameters)
src/graphlime_rdf/
  data/           deterministic RDF loader, leakage blocklist, synthetic generator
  features/       vocabulary + spaces A and B
  models/         R-GCN
  explain/        hsic, hsic_lasso, graphlime, gnnexplainer baseline
  evaluate/       fidelity±, stability, agreement
  reporting/      JSONL → tables/figures, README/report injection
  training.py     seeded runs, manifests, checkpoint bundles
  pipeline.py     validated ExplanationRecords → results/*.jsonl
  cli.py          train | evaluate | repro | tables | readme | load-and-explain
tests/            85%+ coverage gate; hypothesis property tests for the math
docs/             api-notes, architecture (mermaid), method, decision log (ADRs)
results/          JSONL records + generated tables + figures
runs/final/       training audit trail (manifest, config, epochs.csv per run)
checkpoints/      one self-contained best bundle per dataset
notebooks/        qualitative examples (.py source + executed .ipynb + .html)
report/           relazione.md — full project report
```

## Quickstart

```bash
uv sync                 # exact pinned environment (uv.lock)
just check              # lint + mypy --strict + fast tests
just synthetic          # the ground-truth gate
just repro              # full deterministic reproduction (trains + evaluates + tables)
```

### Use the committed checkpoints (no training)

```bash
python -m graphlime_rdf.cli load-and-explain \
    --checkpoint checkpoints/aifb_best.pt --node 5000 --top 5
```

prints the model's prediction for the node and its top predicates, e.g.
`β=0.37  in:http://swrc.ontoware.org/ontology#carriesOut`. The bundle embeds
its manifest (config, seed, git commit, library versions, vocabulary hash)
and is verified on load.

## Documentation

- [`docs/method.md`](docs/method.md) — the math as implemented
- [`docs/architecture.md`](docs/architecture.md) — module map and data flow
- [`docs/api-notes.md`](docs/api-notes.md) — M0 reconnaissance findings
- [`docs/decisions.md`](docs/decisions.md) — ADR-style decision log
- [`report/relazione.md`](report/relazione.md) — the full report
- [`notebooks/qualitative.html`](notebooks/qualitative.html) — worked examples

## Future work

GraphSHAP over the same vocabulary; structural (edge-level) fidelity;
systematic `min_support` sweeps for space B.

## License

MIT
