# Method: GraphLIME with HSIC Lasso, as implemented here

This documents how the method works *in this repository* — the exact
quantities computed, in the notation of the code.

## 1. The interpretable feature space

RDF entity classification usually feeds the R-GCN one-hot entity ids: the
model works, but its input dimensions mean nothing. Here the input is an
**explicit interpretable feature matrix** X (plan §6):

- **Space A** (`features/predicate.py`): column `out:<p>` = number of
  outgoing `<p>` edges of the node (indicator if `binary`), `in:<p>` likewise
  for incoming edges.
- **Space B** (`features/predicate_object.py`): column `out:<p>=<o>` = node
  has predicate `<p>` toward the *specific* term `<o>`; pairs on fewer than
  `min_support` distinct nodes are pruned.

The R-GCN consumes space A as its node features, so a feature that explains
the model is *by construction* a readable statement about the graph
("has publications", "is author of…"). Vocabularies are sorted and
hash-stable (`features/vocabulary.py`).

## 2. HSIC (`explain/hsic.py`)

The Hilbert–Schmidt Independence Criterion measures dependence between two
samples through kernels. With RBF kernels K (on a feature) and L (on the
model output), centering matrix H = I − 11ᵀ/n:

    HSIC(K, L) = trace(K H L H) / (n − 1)²

Empirical HSIC is symmetric, non-negative for PSD kernels, invariant under a
joint permutation of both samples, and — crucially — detects **non-linear**
dependence (`y = x²` with symmetric x has Pearson ≈ 0 but clearly positive
HSIC; asserted in `tests/test_hsic.py`). Bandwidths use the median heuristic.

## 3. HSIC Lasso (`explain/hsic_lasso.py`)

For features x₁…x_d and output y, build the centered, Frobenius-normalised
kernels K̄₁…K̄_d and L̄ and solve

    min_{β ≥ 0}  ½‖vec(L̄) − Σₖ βₖ vec(K̄ₖ)‖²  +  ρ‖β‖₁

with scikit-learn's `Lasso(positive=True)`. Expanding the square shows the
objective rewards features with high HSIC(xₖ, y) and penalises *redundant*
features via HSIC(xₖ, xₗ) — a duplicated feature splits its weight instead
of doubling it (asserted). β is sparse, non-negative, and indexed by the
vocabulary.

## 4. GraphLIME (`explain/graphlime.py`)

To explain node v: take the nodes of its k-hop subgraph (target included) as
the local samples; X = their interpretable features, Y = the model's
predicted probability vectors; run HSIC Lasso. β then says *which named
predicates* the prediction depends on, locally.

Guards:
- neighbourhoods smaller than `min_neighborhood` are **refused** with a
  reason (counted in `results/tables/refusals.md` — a finding, not an error);
- neighbourhoods above `max_neighborhood` are subsampled deterministically
  (RNG seeded by the node id, target kept);
- feature columns constant across the samples get β = 0 exactly and are
  skipped in the solver.

## 5. Evaluation (`evaluate/`)

- **Fidelity±** (`fidelity.py`): column masking on X. fidelity+ = drop in the
  predicted-class probability after zeroing the top-k columns (higher =
  the explanation found necessary features); fidelity− = drop when keeping
  *only* the top-k (lower = they suffice). Every k has a random-k control.
- **Stability** (`stability.py`): Jaccard@k of top-k feature sets across
  training seeds and across hops ∈ {1,2,3}.
- **Agreement** (`agreement.py`): rankings collapsed to predicate URIs —
  space A vs space B, and GraphLIME vs a GNNExplainer attribute-mask
  baseline (`explain/baseline.py`, ADR-007).

## 6. The synthetic gate (`data/synthetic.py`, plan M6)

Before any real-data claim: on a generated graph where class 1 ⟺ the entity
has predicate p₇, GraphLIME must rank `out:syn:p7` first for ≥95% of
explained nodes at 0% noise, and keep it top-3 under correlated distractor
predicates. `just synthetic` runs the gate.
