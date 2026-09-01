# Decision log (ADR-style)

Non-trivial decisions, dated, 3–6 lines each. Part of the audit trail.

## ADR-001 (2026-08-31) — Parse raw RDF ourselves instead of using PyG's processed tensors

PyG's `Entities.process()` discards predicate URIs and orders relations/labels
with `set`-iteration tie-breaking that depends on `PYTHONHASHSEED`
(see `docs/api-notes.md` Q1). Readable names and cross-process determinism are
core deliverables, so `data/loader.py` parses `<name>_stripped.nt.gz` and the
TSV splits directly with lexicographically sorted indexing. PyG is used only
to download/extract the identical benchmark archive.

## ADR-002 (2026-08-31) — Leakage blocklist asserts absence rather than removing

The published `_stripped` dumps already exclude the target-revealing
predicates (AIFB: `employs`, `affiliation`; MUTAG: `isMutagenic` — verified
empirically). `data/leakage.py` keeps the explicit per-dataset blocklist and
the loader asserts none of them occur, failing loudly if upstream files ever
change. Near-label predicates the benchmark convention keeps (e.g. MUTAG
`amesTestPositive`) stay in — explainer behaviour on them is a finding.

## ADR-003 (2026-08-31) — Implement fidelity ourselves; PyG helpers are edge-mask-coupled

`torch_geometric.explain.metric.fidelity` is bound to the
`Explainer`/`Explanation` pair and edge/node masking. Our locked design
defines fidelity as column masking on the interpretable feature matrix, so
`evaluate/fidelity.py` implements fidelity± directly (we control semantics,
and `edge_type` forwarding stops being an issue outside the GNNExplainer
baseline).

## ADR-009 (2026-08-31) — Synthetic gate uses out-only features so column masking measures the mechanism

With [out,in] features the R-GCN classified synthetic entities through the
*object* node's `in:p7` count carried by the p7-typed message: masking the
entity's own `out:p7` column left predictions intact (fid+ gap ≈ 0.003,
noise-level) even though GraphLIME correctly ranked `out:p7` first.
Diagnosed via `scripts/debug_fidelity.py`. Fix is in the testbed, not the
threshold: `configs/synthetic.yaml` now uses `directions: [out]`, making
object rows all-zero so the class signal must flow through the entity's own
feature column — exactly what column-masking fidelity measures. Post-fix:
class-1 fid+ 0.20 vs random 0.06; gate and margin (0.05) unchanged, passing.

## ADR-007 (2026-08-31) — GNNExplainer baseline uses the attribute mask, not edge masks

PyG's explainer installs a full-graph edge mask on every MessagePassing
module, but `RGCNConv` runs one propagate per relation on edge *subsets* —
the sizes can never match (verified: AssertionError inside
`message_passing.py`; known upstream limitation). Rather than reimplementing
R-GCN or silently dropping the baseline, GNNExplainer learns its **attribute
mask** over the same interpretable feature matrix, scoring exactly the
features GraphLIME scores; rankings are aggregated per predicate for the
agreement metric. Tighter comparison, honest mechanics.

## ADR-008 (2026-08-31) — GraphLIME neighbourhood cap with deterministic subsampling

k-hop neighbourhoods of hub nodes reach thousands of nodes and the HSIC Lasso
design matrix has n² rows. Above `max_neighborhood` (default 200) we
subsample without replacement with an RNG seeded by the target node id
(target always kept) — deterministic per node across runs. Constant feature
columns within the sample are skipped in the solver (their β is provably 0)
and scattered back, keeping output length equal to the vocabulary size.

## ADR-006 (2026-08-31) — Hyperparameter choice from the fixed M3 grid

Grid explored (scratch runs, `scripts/grid_m3.py` + `scripts/probe_m3.py`):
directions {[out], [out,in]} × hidden {16, 32} × binary {true, false} ×
weight_decay {5e-4, 0} × epochs {50, 200}. Binary indicators cap AIFB at
~0.82; predicate **counts** (still fully interpretable: "number of
publications") with directions [out,in], hidden 32, epochs 200, wd 0 reach
AIFB ≈ 0.93 and MUTAG ≈ 0.75 (probe seeds). Chosen for both datasets and
frozen in `configs/{aifb,mutag}.yaml`; 5-seed DoD gates enforced by the slow
tests in `tests/test_training.py`.

## ADR-005 (2026-08-31) — Canonicalise blank nodes with rdflib RGDA1

AIFB contains 152 blank nodes whose ids rdflib regenerates on every parse,
which silently broke cross-process determinism (caught by
`test_loader_deterministic_across_calls`). The loader now runs
`rdflib.compare.to_canonical_graph` (content-derived RGDA1 labeling) whenever
bnodes are present, verified byte-identical across `PYTHONHASHSEED` ∈
{0, 1, 12345} via `scripts/hash_graph.py`.

## ADR-004 (2026-08-31) — Python 3.13.5, torch 2.13.0 (CPU), PyG 2.8.0.post1

Newest stable stack available via `uv` on this machine; all pins recorded in
`uv.lock` and `docs/api-notes.md` Q6. CPU-only suffices for AIFB/MUTAG scale.
`just render` installs the `graphlime-rdf` ipykernel spec (idempotent) so the
notebook pipeline works headless from a fresh clone.

## ADR-010 (2026-08-31) — README carries no numbers; results live in the report

The README had duplicated every generated table and figure, which made it a
second place for results to drift and buried the orientation material a reader
actually needs from a landing page. It is now prose only: what the project
does, where to find what, and how to run it, closing with links to
`report/relazione.pdf` and `report/presentazione.pdf` plus the bibliography.
`reporting/readme.py` therefore injects into `report/relazione.md` alone;
`inject()` still raises on a document missing its markers, so the report can
never silently stop being refreshed.

## ADR-011 (2026-09-01) — The qualitative notebook becomes the live demo

`notebooks/qualitative.py` printed sentence-form explanations and nothing
else, which made it a results appendix rather than something anyone could
present. It now walks the project's whole argument in order — the RDF entity
with no feature vector, the manufactured feature space, the vocabulary-hash
identity between model input and explanation output, one explanation end to
end, fidelity/stability/baseline checks on that same node, the synthetic gate
replayed live, the qualitative gallery, and finally the generated tables and
figures — so that it can be run in front of an audience before the slide deck.
Two constraints kept it honest: it composes only functions that already exist
in `src/` (the notebook adds display glue, never method), and it retrains
nothing on AIFB/MUTAG, loading `checkpoints/<dataset>_best.pt` instead. The
worked example is the *first* AIFB test entity, stated as such, not a
cherry-picked one; where its output is uninformative — a node saturated at
p = 1.00 flattens fidelity+, and GNNExplainer's mask ties most predicates at
zero — the notebook says so and defers to the aggregates in §10 rather than
choosing a friendlier node. Total runtime ≈ 40 s, of which ~22 s is the
baseline cell.

## ADR-012 (2026-09-01) — `ipywidgets` in the dev group

Importing `torch_geometric` pulls in `tqdm.auto`, which warns `IProgress not
found` on every first cell run inside a Jupyter kernel because the widget
frontend was missing from the environment. The warning is cosmetic but it is
the first thing an audience sees in a demo. `ipywidgets>=8.1` is now a dev
dependency; no source change was needed.

