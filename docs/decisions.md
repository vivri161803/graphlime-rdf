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
