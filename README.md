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
### Classification accuracy

| dataset | test accuracy (mean ± std, 5 seeds) | per-seed | best seed | best accuracy | checkpoint sha256 |
|---|---|---|---|---|---|
| AIFB | 0.8833 ± 0.0534 | 0.9167, 0.9444, 0.8056, 0.8889, 0.8611 | 1 | 0.9444 | `8f69360a52b0` |
| MUTAG | 0.7559 ± 0.0305 | 0.7794, 0.7206, 0.7353, 0.7500, 0.7941 | 4 | 0.7941 | `20de0d5614ec` |

### Explanation fidelity (column masking, random-k control)

| dataset | k | fidelity+ | random+ | fidelity− | random− | nodes |
|---|---|---|---|---|---|---|
| AIFB | 1 | 0.0176 | 0.0129 | 0.7145 | 0.6767 | 36 |
| AIFB | 2 | 0.0162 | 0.0066 | 0.7386 | 0.7556 | 36 |
| AIFB | 5 | 0.1079 | 0.0520 | 0.7160 | 0.6085 | 36 |
| AIFB | 10 | 0.1829 | 0.0735 | 0.6338 | 0.6184 | 36 |
| MUTAG | 1 | 0.0789 | -0.0013 | 0.5397 | 0.3151 | 68 |
| MUTAG | 2 | 0.1224 | 0.0205 | 0.3068 | 0.2744 | 68 |
| MUTAG | 5 | 0.1627 | 0.0475 | 0.1987 | 0.3352 | 68 |
| MUTAG | 10 | 0.0999 | 0.1592 | 0.2299 | 0.3462 | 68 |

### Explanation stability (Jaccard@5)

| dataset | comparison | pair | mean Jaccard@5 | pairs |
|---|---|---|---|---|
| AIFB | hops | hops=1 vs hops=2 | 0.0413 | 21 |
| AIFB | hops | hops=2 vs hops=3 | 0.0585 | 36 |
| AIFB | seeds | seed=0 vs seed=1 | 0.2443 | 36 |
| AIFB | seeds | seed=1 vs seed=2 | 0.5162 | 36 |
| AIFB | seeds | seed=2 vs seed=3 | 0.4547 | 36 |
| AIFB | seeds | seed=3 vs seed=4 | 0.4239 | 36 |
| MUTAG | hops | hops=1 vs hops=2 | 0.0103 | 68 |
| MUTAG | hops | hops=2 vs hops=3 | 0.0696 | 68 |
| MUTAG | seeds | seed=0 vs seed=1 | 0.3593 | 68 |
| MUTAG | seeds | seed=1 vs seed=2 | 0.1709 | 68 |
| MUTAG | seeds | seed=2 vs seed=3 | 0.3179 | 68 |
| MUTAG | seeds | seed=3 vs seed=4 | 0.7475 | 68 |

### Feature-space sensitivity and baseline agreement

| dataset | comparison | mean Jaccard@5 (predicates) | nodes |
|---|---|---|---|
| AIFB | GraphLIME: space A vs space B | 0.2843 | 36 |
| AIFB | GraphLIME vs GNNExplainer | 0.1456 | 36 |
| MUTAG | GraphLIME: space A vs space B | 0.0029 | 68 |
| MUTAG | GraphLIME vs GNNExplainer | 0.3153 | 68 |

### Refused explanations

| dataset | explained | refused | refusal rate | reasons |
|---|---|---|---|---|
| AIFB | 36 | 0 | 0.0000 | — |
| MUTAG | 68 | 0 | 0.0000 | — |

![Fidelity+ vs k against the random-k control](results/figures/fidelity_curve.png)
![Stability heatmap across seeds and hops](results/figures/stability_heatmap.png)
![Feature-space (A vs B) and baseline agreement](results/figures/agreement_comparison.png)
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
