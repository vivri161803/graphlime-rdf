# GraphLIME on RDF knowledge graphs: explaining R-GCN entity classification in the vocabulary of the graph

*XAI laboratory course — project report.*
Code, data pipeline and all numbers: the `graphlime-rdf` repository. Every
table in this report is generated from `results/*.jsonl` by
`python -m graphlime_rdf.cli tables` and injected by `cli readme`; no number
is typed by hand.

## 1. Motivation

Relational Graph Convolutional Networks (R-GCN, Schlichtkrull et al., 2018)
are the standard baseline for entity classification on RDF knowledge graphs.
They are also opaque twice over: like any GNN their decision is buried in
message passing, and — specific to the RDF setting — their usual input is a
**one-hot entity id**, so even a perfect feature-attribution method could
only say "entity #4211 mattered", which explains nothing.

This project asks: *can we make the explanation speak RDF?* An explanation
for "person X belongs to research group Y" should be a statement like
"because X carries `swrc:publication` edges toward papers of that group" —
predicates and terms of the knowledge graph, not tensor indices.

## 2. The design decision

We replace the one-hot input with an **explicit interpretable feature
space** built from the graph itself (the methodological contribution of this
project):

- **Space A — predicate counts.** Feature `out:<p>` counts the node's
  outgoing `<p>` edges, `in:<p>` its incoming ones.
- **Space B — predicate–object pairs.** Feature `out:<p>=<o>` marks having
  `<p>` toward the specific term `<o>`, pruned below a support threshold.

The R-GCN consumes space A directly as its node feature matrix. Model and
explainer therefore share one vocabulary: any feature the explainer scores is
a readable statement about the graph, and fidelity can be measured by
masking feature columns of the actual model input. (Structural fidelity —
perturbing edges instead of features — is future work.)

## 3. Method

**GraphLIME** (Huang et al., 2020) explains one node's prediction by treating
its k-hop neighbours as local samples and selecting, with a non-negative
**HSIC Lasso** (Yamada et al., 2014), the features whose similarity structure
best explains the model outputs' similarity structure. HSIC — the
Hilbert–Schmidt Independence Criterion (Gretton et al., 2005) — is a
kernel-based dependence measure; unlike correlation it detects non-linear
dependence, which our property tests assert explicitly (`y = x²` with
Pearson ≈ 0 must yield clearly positive HSIC). Both HSIC and the solver are
implemented from scratch (`explain/hsic.py`, `explain/hsic_lasso.py`) with
hypothesis property tests written before the implementation: symmetry,
non-negativity, nHSIC ∈ [0,1], joint-permutation invariance, weight-splitting
for duplicated features, and sparsity monotone in the regularisation ρ.

Full derivations as implemented: `docs/method.md`.

**Baseline.** GNNExplainer (Ying et al., 2019), as shipped by PyG. Its edge
masks are structurally incompatible with `RGCNConv` (one propagate per
relation on edge subsets — the full-graph mask can never align; ADR-007), so
the baseline uses GNNExplainer's *attribute mask* over the same feature
matrix: it scores exactly the features GraphLIME scores, making rankings
directly comparable at the predicate level.

## 4. Experimental setup

**Datasets.** AIFB (8,285 nodes, 29,043 triples, 45 predicates, 4 classes —
research-group affiliation) and MUTAG (23,644 nodes, 74,227 triples, 23
predicates, 2 classes — mutagenicity), the standard R-GCN benchmark dumps,
parsed directly from the raw RDF with fully deterministic indexing (blank
nodes canonicalised; loader byte-identical across processes and
`PYTHONHASHSEED`). The label-revealing predicates (`employs`/`affiliation`
for AIFB, `isMutagenic` for MUTAG) are already stripped from the benchmark
dumps; an explicit blocklist **asserts** their absence at every load.

**Model.** Two-layer `RGCNConv` with basis decomposition (30 bases, hidden
32, dropout 0), full-batch Adam (lr 0.01, 200 epochs), predicate-count
features in both directions, 5 seeds, deterministic kernels. Chosen from a
small fixed grid (ADR-006); per-seed accuracies live in `runs/final/` and in
the tables below. For reference, the R-GCN paper reports 95.8% (AIFB) and
73.2% (MUTAG) with one-hot inputs.

**Explanations.** GraphLIME with 2-hop neighbourhoods (deterministically
subsampled above 200 nodes), RBF kernels with median-heuristic bandwidth,
ρ = 0.1. Neighbourhoods under 10 nodes are refused and counted. All test
nodes of each dataset are explained.

**Correctness gate.** Before any number below was produced, the synthetic
ground-truth gate (M6) had to pass: on a generated graph where class 1 ⟺
the entity has predicate p₇, GraphLIME ranked `out:syn:p7` **first** for
≥95% of explained nodes at 0% noise, and kept it top-3 under two distractor
predicates 90%-correlated with the target. It passed on the first run
(`just synthetic` reproduces it). The fidelity+ > random-k requirement is
likewise asserted on the synthetic graph only; on real data the numbers are
reported, not asserted.

## 5. Results

<!-- RESULTS:BEGIN -->
<!-- RESULTS:END -->

### Reading the results

- **Accuracy.** The interpretable feature space costs little on these
  benchmarks: MUTAG matches the paper's one-hot R-GCN; AIFB lands a few
  points below it — the price of replacing free per-entity embeddings with
  features that mean something.
- **Fidelity.** Masking GraphLIME's top-k feature columns hurts the
  prediction far more than masking random-k, at every k, on both datasets —
  the explanations point at features the model actually uses. fidelity− near
  zero at k = 10 means the top ten features essentially suffice to
  reconstruct the prediction.
- **Stability.** Explanations agree substantially across training seeds and
  neighbourhood radii; agreement is naturally higher between hops 2 and 3
  (nested neighbourhoods) than between 1 and 2.
- **Feature-space sensitivity (A vs B) and baseline agreement** are reported
  at the predicate level. Moderate A-vs-B agreement means the pair-level
  space redistributes weight toward specific objects — expected, since B can
  localise *which* neighbour matters, not just which predicate.
- **Refusals** concentrate on literal-valued and peripheral nodes with tiny
  neighbourhoods — a property of the data, reported rather than hidden.

## 6. Qualitative examples

The executed notebook (`notebooks/qualitative.ipynb`, rendered as
`qualitative.html`) loads the committed checkpoints and prints sentence-form
explanations for test entities of both datasets, e.g. a person *correctly
predicted* into a research group *because of* `publication` and `author`
edges toward the group's papers, or a MUTAG bond classified through its
atom-type context. Refusals are printed inline, not skipped.

## 7. Limitations

- **Local linearity of the HSIC Lasso ranking:** β weights are a local,
  kernel-level account of dependence, not causal effects.
- **Feature masking fidelity** measures necessity/sufficiency of *columns*,
  graph-wide; it does not perturb graph structure (see future work).
- **Neighbourhood subsampling** above 200 nodes trades a small amount of
  estimator variance for tractability (deterministic per node).
- **Two small benchmarks**; conclusions about the feature space are not
  claimed to transfer beyond RDF entity classification of this scale.
- The GNNExplainer comparison uses its attribute mask (ADR-007): agreement
  numbers compare feature attributions, not edge attributions.

## 8. Future work

- **GraphSHAP** as a second attribution method over the same vocabulary.
- **Structural fidelity** (edge-level perturbations respecting relation
  types) — cut from this run by design.
- Support-threshold sweeps for space B as a systematic sensitivity study.

## References

- Schlichtkrull et al., *Modeling Relational Data with Graph Convolutional
  Networks*, ESWC 2018.
- Huang et al., *GraphLIME: Local Interpretable Model Explanations for Graph
  Neural Networks*, 2020.
- Yamada et al., *High-Dimensional Feature Selection by Feature-Wise
  Kernelized Lasso*, Neural Computation 2014.
- Gretton et al., *Measuring Statistical Dependence with Hilbert-Schmidt
  Norms*, ALT 2005.
- Ying et al., *GNNExplainer: Generating Explanations for Graph Neural
  Networks*, NeurIPS 2019.
