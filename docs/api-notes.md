# M0 — API reconnaissance notes

Date: 2026-08-31. All findings verified empirically against the installed
versions (see §6) via `scripts/recon_m0.py` and `scripts/recon_m0_data.py`.

## Q1 — Does `torch_geometric.datasets.Entities` expose predicate names?

**No — but the raw RDF dumps remain on disk, and we recover names from them.**

`Entities.process()` parses `data/<name>/raw/<name>_stripped.nt.gz` with
`rdflib` itself, then discards all URIs: the processed `Data` object carries
only integer `edge_type` ids. Two additional problems with reusing PyG's
processing:

1. **Nondeterministic tie-breaking.** Relations are ordered by
   `sorted(set(g.predicates()), key=lambda p: -freq[p])`. Ties keep the
   iteration order of a `set` of `URIRef` (a `str` subclass), which depends on
   `PYTHONHASHSEED`. A mapping rebuilt in a different process could disagree
   with the cached `data.pt`.
2. **Nondeterministic label ids.** `labels_dict` is built from a `set` of
   label strings — same hazard.

**Consequence (locked decision, see `docs/decisions.md` ADR-001):** we do not
use PyG's processed tensors at all. `graphlime_rdf.data.loader` parses the raw
`.nt.gz` and the train/test TSVs itself with fully deterministic
(lexicographically sorted) node / relation / label indexing. We reuse PyG only
for `download_url` + `extract_tar` of the same archive
(`https://data.dgl.ai/dataset/{aifb,mutag}.tgz`), so the underlying benchmark
files are byte-identical to the standard ones.

Raw file inventory per dataset: `<name>_stripped.nt.gz` (graph),
`trainingSet.tsv`, `testSet.tsv`, `completeDataset.tsv`. TSV headers:
AIFB → node column `person`, label column `label_affiliation`;
MUTAG → node column `bond`, label column `label_mutagenic`.

## Q2 — Does PyG strip the leakage relations?

**PyG itself strips nothing**, but the benchmark dumps are the `_stripped`
variants published for the R-GCN paper, and empirically the target-revealing
predicates are already absent:

- AIFB (29,043 triples, 8,285 nodes, 45 predicates): no predicate URI
  containing `employs` or `affiliation`.
- MUTAG (74,227 triples, 23,644 nodes, 23 predicates): no predicate URI
  containing `isMutagenic`.

Node counts match the published stats exactly (AIFB 8,285 / MUTAG 23,644).
We still keep an explicit blocklist in `data/leakage.py` and **assert absence
at load time** — if upstream ever swaps in unstripped dumps, the loader fails
loudly instead of leaking labels. Near-label predicates that legitimately
remain in MUTAG (e.g. `carcinogenesis#amesTestPositive`) are *not* removed:
the benchmark convention keeps them, and whether the explainer surfaces them
is a finding, not leakage.

## Q3 — Which explainers ship in `torch_geometric.explain.algorithm`?

`AttentionExplainer, CaptumExplainer, DummyExplainer, GNNExplainer,
GraphMaskExplainer, PGExplainer`.

`GNNExplainer` is present → used as the baseline. **GraphLIME is not shipped**
→ implemented from scratch in `explain/graphlime.py` as planned.

## Q4 — Do the fidelity metric helpers fit our needs?

They exist (`torch_geometric.explain.metric.fidelity`,
`characterization_score`, `fidelity_curve_auc`, …) but
`fidelity(explainer, explanation)` is coupled to the `Explainer` /
`Explanation` object pair and to *edge/node masking* semantics. Our locked
design (plan §6) defines fidelity as **column masking on the interpretable
feature matrix `x`**, and `RGCNConv` needs `edge_type` forwarded on every
call. **We implement fidelity ourselves** in `evaluate/fidelity.py`
(~20 lines) so we control the semantics; the wrapper-closure trick for
`edge_type` is only needed for the GNNExplainer baseline (verified:
`Explainer(model, ...)` passes `**kwargs` through to `model.forward`, and
GNNExplainer accepts a fixed `edge_type` kwarg).

## Q5 — jupytext + nbconvert headless pipeline

Verified working:
`jupytext --to ipynb --execute --set-kernel graphlime-rdf x.py -o x.ipynb`
followed by `jupyter nbconvert --to html x.ipynb`. Requires a registered
kernelspec; `python -m ipykernel install --user --name graphlime-rdf` is
idempotent and is part of `just render` so a fresh clone renders without
manual setup.

## Q6 — Pinned versions (`uv.lock`, Python 3.13.5)

| package | version |
|---|---|
| torch | 2.13.0 |
| torch-geometric | 2.8.0.post1 |
| numpy | 2.5.2 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| pandas | 3.0.5 |
| pydantic | 2.13.5 |
| rdflib | 7.6.0 |
| typer | 0.27.2 |
| jupytext | 1.19.5 |
| nbconvert | 7.17.1 |

`torch.use_deterministic_algorithms(True)` verified importable and enabled
without error on this build (CPU).
