# graphlime-rdf

**GraphLIME explanations for R-GCN entity classification on RDF knowledge
graphs — expressed in the vocabulary of the graph itself.**

XAI laboratory course project. This file describes *what the project is* and
*where everything lives*. The results, their interpretation and the full
argument are in the report — see [Deliverables](#deliverables) at the end.

## What the project does

An R-GCN normally classifies RDF entities from **one-hot entity ids**:
accurate, but unexplainable in principle — no attribution method can make
"entity #4211" meaningful to a reader.

The deeper obstacle is that **GraphLIME cannot be applied to an RDF graph as
published**. Its mechanism operates on a node *feature matrix*: it takes the
feature vectors of the target's neighbours and asks, via kernels, which
feature dimensions the model's output depends on. RDF entities have no
feature vectors at all — every piece of information sits on a typed edge — so
the explainer's input is not merely impoverished, it is undefined.

This project's contribution is to **manufacture that missing feature space
from the graph's own predicates**:

- **Space A — predicate counts.** `out:<p>` counts the node's outgoing `<p>`
  edges, `in:<p>` its incoming ones.
- **Space B — predicate–object pairs.** `out:<p>=<o>` marks having `<p>`
  toward a specific term `<o>`, pruned below a support threshold.

The R-GCN then consumes space A *in place of* the one-hot input, so model and
explainer share one vocabulary. Two things follow: every explanation is a
ranked list of **named predicates** a domain expert can check directly (an
integer index where a name belongs is, by project rule, a bug), and
**fidelity becomes measurable at all**, as masking feature columns of the
actual model input.

Explanations are produced by a from-scratch **GraphLIME** (HSIC Lasso,
kernels and solver implemented in this repository) and evaluated on fidelity,
stability and agreement, with GNNExplainer as a baseline. Correctness is
gated on a synthetic graph with a planted ground-truth predicate: that gate
had to pass before any real-data number was produced, property tests written
before the implementation pin down the HSIC maths, and every reported number
is generated from validated JSONL records carrying the config hash, seed and
git commit that produced it.

Datasets are the standard R-GCN benchmarks, AIFB and MUTAG.

## Where to find what

Start here depending on what you are after:

| If you want to… | Go to |
|---|---|
| read the full argument and the results | [`report/relazione.pdf`](report/relazione.pdf) |
| get the project in 26 slides | [`report/presentazione.pdf`](report/presentazione.pdf) |
| see the maths exactly as implemented | [`docs/method.md`](docs/method.md) |
| understand the module layout and data flow | [`docs/architecture.md`](docs/architecture.md) |
| know why something was done that way | [`docs/decisions.md`](docs/decisions.md) |
| see explanations for real entities | [`notebooks/qualitative.html`](notebooks/qualitative.html) |
| reproduce every number | [`just repro`](#reproducing-everything) |
| explain a node without training | [`load-and-explain`](#use-the-committed-checkpoints-no-training) |

## Repository structure

```
configs/            experiment YAMLs — the single source of hyperparameters
                      aifb.yaml, mutag.yaml, synthetic.yaml

src/graphlime_rdf/
  config.py         typed config objects; no magic number lives outside here
  data/
    loader.py       deterministic RDF parser (canonicalised blank nodes)
    leakage.py      blocklist asserting label-revealing predicates are absent
    synthetic.py    generator for the ground-truth gate
  features/
    vocabulary.py   sorted, hash-stable feature vocabularies
    predicate.py    space A — predicate counts
    predicate_object.py   space B — predicate–object pairs
  models/rgcn.py    two-layer R-GCN with basis decomposition
  explain/
    hsic.py         HSIC estimator (from scratch)
    hsic_lasso.py   non-negative HSIC Lasso solver
    graphlime.py    neighbourhood sampling, refusals, β → predicate names
    baseline.py     GNNExplainer attribute-mask baseline
  evaluate/
    fidelity.py     fidelity± by column masking, with random-k controls
    stability.py    Jaccard@k across seeds and hops
    agreement.py    space A vs B, and GraphLIME vs the baseline
  reporting/
    tables.py       JSONL → markdown tables
    figures.py      JSONL → figures
    readme.py       marker-based injection of results into the report
  training.py       seeded runs, manifests, checkpoint bundles
  pipeline.py       validated ExplanationRecords → results/*.jsonl
  cli.py            train | explain | evaluate | tables | readme |
                    load-and-explain | repro

tests/              85% coverage gate; hypothesis property tests for the maths
                    (test_hsic.py, test_hsic_lasso.py) and the ground-truth
                    gate (test_synthetic_gate.py)

docs/               method.md, architecture.md, decisions.md (ADRs),
                    api-notes.md
report/             relazione.md + relazione.pdf — the full report
                    presentazione.tex + presentazione.pdf — the slide deck
notebooks/          qualitative examples (.py source, executed .ipynb, .html)

results/            validated JSONL records, generated tables/ and figures/
runs/final/         training audit trail (manifest, config, epochs.csv per run)
checkpoints/        one self-contained best bundle per dataset
scripts/            one-off reconnaissance and grid-search helpers
```

Three directories are contractual: `results/` holds validated records plus
the tables and figures generated from them, `runs/` is the training audit
trail, and `checkpoints/` holds best weights only.

## How to use the repository

### Setup

```bash
uv sync
```

Reconstructs the exact pinned environment from `uv.lock` (Python 3.13, CPU-only
torch and PyG). The task runner is [`just`](https://github.com/casey/just);
every quality gate lives in the `justfile`.

### Everyday commands

```bash
just check
```

Lint, `mypy --strict` and the fast tests with the coverage gate — green before
every commit.

```bash
just synthetic
```

The ground-truth gate: on a generated graph where class 1 ⟺ the entity has
predicate `p₇`, GraphLIME must rank that predicate first for ≥95% of explained
nodes, and keep it top-3 under correlated distractors.

### Use the committed checkpoints (no training)

```bash
python -m graphlime_rdf.cli load-and-explain --checkpoint checkpoints/aifb_best.pt --node 5000 --top 5
```

Prints the model's prediction for the node and its top predicates, e.g.
`β=0.37  in:http://swrc.ontoware.org/ontology#carriesOut`. The bundle embeds
its manifest (config, seed, git commit, library versions, vocabulary hash) and
is verified on load.

### Reproducing everything

```bash
just repro
```

Trains both datasets over 5 seeds, explains every test entity, evaluates, and
regenerates `results/tables/` and `results/figures/`. Numbers in the report are
then injected from those tables by `just readme`; none is typed by hand.

```bash
just render
```

Re-executes the qualitative notebook and renders it to HTML.

```bash
just pdf
just slides
```

Rebuild the report PDF and the slide deck (needs pandoc and a TeX
distribution).

## Deliverables

- **[`report/relazione.pdf`](report/relazione.pdf)** — the full project
  report: motivation, the design decision, method with derivations,
  experimental setup, results with their interpretation, and limitations.
  Markdown source: [`report/relazione.md`](report/relazione.md).
- **[`report/presentazione.pdf`](report/presentazione.pdf)** — 26-slide
  executive summary of the same material. Source:
  [`report/presentazione.tex`](report/presentazione.tex).

The report is the thorough treatment; the slides summarise it and are not a
substitute for it.

## Bibliography

- Schlichtkrull, M., Kipf, T., Bloem, P., van den Berg, R., Titov, I., and
  Welling, M. *Modeling Relational Data with Graph Convolutional Networks*.
  ESWC, 2018.
- Huang, Q., Yamada, M., Tian, Y., Singh, D., Yin, D., and Chang, Y.
  *GraphLIME: Local Interpretable Model Explanations for Graph Neural
  Networks*. 2020.
- Yamada, M., Jitkrittum, W., Sigal, L., Xing, E. P., and Sugiyama, M.
  *High-Dimensional Feature Selection by Feature-Wise Kernelized Lasso*.
  Neural Computation, 2014.
- Gretton, A., Bousquet, O., Smola, A., and Schölkopf, B. *Measuring
  Statistical Dependence with Hilbert-Schmidt Norms*. ALT, 2005.
- Ying, R., Bourgeois, D., You, J., Zitnik, M., and Leskovec, J.
  *GNNExplainer: Generating Explanations for Graph Neural Networks*.
  NeurIPS, 2019.

## License

MIT — see [`LICENSE`](LICENSE).
