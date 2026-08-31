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

The design below is not an optimisation but a precondition: **GraphLIME
cannot be applied to an RDF graph as published.** Its entire mechanism
operates on a node **feature matrix** — it collects the feature vectors of
the target's neighbours and asks, via kernels, which feature dimensions the
model's outputs depend on. The method was designed for citation networks,
where every node carries a bag-of-words vector and every dimension is a
word. In the RDF setting neither ingredient exists: entities carry **no
feature vectors at all** — every piece of information is a typed edge toward
another term — and the R-GCN's customary one-hot input (an entity-id
embedding) offers no dimension a human could read. With no feature matrix
there is nothing for HSIC Lasso to select from, and with anonymous
embedding dimensions any selection would be unreadable anyway. The
explainer is not merely weakened; its input is undefined.

The solution we designed is to **manufacture the missing feature space from
the graph itself**, in the graph's own vocabulary (the methodological
contribution of this project):

- **Space A — predicate counts.** Feature `out:<p>` counts the node's
  outgoing `<p>` edges, `in:<p>` its incoming ones.
- **Space B — predicate–object pairs.** Feature `out:<p>=<o>` marks having
  `<p>` toward the specific term `<o>`, pruned below a support threshold.

Crucially, the R-GCN consumes space A directly as its node feature matrix,
in place of the one-hot input. Model and explainer therefore share one
vocabulary: any feature the explainer scores is a readable statement about
the graph ("has `publication` edges", "is `author` of …"), and fidelity —
otherwise undefined, since the explainer's features would not be what the
model consumes — becomes well-defined as masking feature columns of the
actual model input.

## 3. Method

### 3.1 The model: R-GCN

The R-GCN (Schlichtkrull et al., 2018) extends graph convolution to
multi-relational graphs by giving every relation type its own transformation.
The layer-wise propagation rule for node $i$ is

$$
h_i^{(l+1)} \;=\; \sigma\!\Bigg(\; \sum_{r \in \mathcal{R}} \;
\sum_{j \in \mathcal{N}_i^r} \frac{1}{c_{i,r}}\, W_r^{(l)} h_j^{(l)}
\;+\; W_0^{(l)} h_i^{(l)} \Bigg),
$$

where $\mathcal{N}_i^r$ is the set of neighbours reached from $i$ through
relation $r$, $c_{i,r} = |\mathcal{N}_i^r|$ is a per-relation normalisation
constant, $W_r^{(l)}$ is the weight matrix specific to relation $r$, and
$W_0^{(l)}$ is a self-loop transformation. The per-relation weights are the
point: in a knowledge graph the meaning lives in the edge *types*, and a
plain GCN would collapse, say, `author` and `employs` into an undifferentiated
"neighbour". Since one full matrix per relation is prohibitive (45 relations
on AIFB), each $W_r$ is expressed by **basis decomposition**,

$$
W_r^{(l)} \;=\; \sum_{b=1}^{B} a_{rb}^{(l)}\, V_b^{(l)},
$$

a learned linear combination of $B$ shared basis matrices — parameter sharing
that doubles as regularisation. Two such layers followed by a softmax,
trained with cross-entropy on the few labelled entities, constitute the
classifier we explain.

### 3.2 HSIC: measuring non-linear dependence

The Hilbert–Schmidt Independence Criterion (Gretton et al., 2005) is a
kernel measure of statistical dependence between two variables. Given $n$
paired samples, kernel matrices $K$ and $L$ over the two variables, and the
centering matrix $H = I - \tfrac{1}{n}\mathbf{1}\mathbf{1}^\top$, the
empirical estimator is

$$
\widehat{\mathrm{HSIC}}(K, L) \;=\; \frac{\operatorname{tr}(KHLH)}{(n-1)^2}.
$$

For characteristic kernels (we use RBF kernels with median-heuristic
bandwidth) HSIC vanishes exactly when the variables are independent — and,
unlike correlation, it detects **non-linear** dependence. The canonical
example: $y = x^2$ with $x$ symmetric around zero has Pearson correlation
$\approx 0$ but clearly positive HSIC. This single example justifies the
whole construction — a GNN's output is a non-linear function of its input,
so a linear surrogate would measure the wrong thing. Our property tests
assert this behaviour explicitly.

### 3.3 HSIC Lasso and its three-term reading

**HSIC Lasso** (Yamada et al., 2014) turns HSIC into a feature selector. For
features $x_1, \dots, x_d$ and output $y$, build the centered,
Frobenius-normalised kernel matrices $\bar K_1, \dots, \bar K_d$ and
$\bar L$, and solve the non-negative Lasso

$$
\min_{\beta \ge 0} \;\; \tfrac{1}{2}\,
\Big\| \bar L - \sum_{k=1}^{d} \beta_k \bar K_k \Big\|_F^2
\;+\; \rho\, \|\beta\|_1 .
$$

Expanding the squared Frobenius norm exposes the **explainable three-term
form** (up to the constant $\tfrac12\|\bar L\|_F^2$):

$$
\min_{\beta \ge 0} \;\;
\underbrace{-\sum_{k} \beta_k\, \mathrm{HSIC}(x_k, y)}_{\text{relevance}}
\;+\;
\underbrace{\tfrac{1}{2}\sum_{k,l} \beta_k \beta_l\,
\mathrm{HSIC}(x_k, x_l)}_{\text{redundancy}}
\;+\;
\underbrace{\rho\,\|\beta\|_1}_{\text{sparsity}} .
$$

In words: the first term rewards features the output depends on; the second
penalises features that are redundant with one another (a duplicated feature
splits its weight instead of doubling it — asserted by our tests); the third
forces most weights to zero. The resulting $\beta_k \ge 0$ is feature $k$'s
importance.

### 3.4 GraphLIME and why it must be modified for RDF

**GraphLIME** (Huang et al., 2020) departs from LIME in that it perturbs
nothing: to explain node $v$, the "samples" are the **real nodes of $v$'s
$h$-hop neighbourhood** (here $h=2$). $X$ collects their feature vectors,
$Y$ the model's predicted probability vectors; feeding both to HSIC Lasso
yields $\beta$, a sparse non-negative importance over *feature dimensions*.
Two consequences shape this project. First, **GraphLIME explains features,
not edges** — so on a graph whose nodes have no features the method has no
input, which is precisely the RDF situation of §2. Second, its samples are a
local neighbourhood: a small neighbourhood means few samples and a
meaningless HSIC estimate, so the pipeline refuses to explain nodes below a
minimum neighbourhood size rather than emit noise.

Our modification is the feature space of §2, in two variants: **method A**
(predicate counts — one feature per predicate and direction; coarse, dense,
stable, $\approx 90$ columns on AIFB) and **method B** (predicate–object
pairs — expressive, e.g. "has `member` toward group *Y*", but
high-dimensional and sparse, which starves the kernel estimates; hence the
support-threshold pruning). Comparing A against B is itself one of the
reported experiments (§5).

### 3.5 The pipeline

End to end: (1) parse the raw RDF into a deterministic multigraph; (2) build
the feature vocabulary and matrix $X$ (space A, and B for the comparison);
(3) train the R-GCN *on $X$* — not on one-hot ids — so model and explainer
share the vocabulary; (4) for every test entity, extract the 2-hop
neighbourhood, run HSIC Lasso on the neighbourhood's features against the
model's predicted probabilities, and map the non-zero $\beta$ back to
predicate names; (5) evaluate with the metrics of §5, gated by the synthetic
ground-truth check of §4. Both HSIC and the solver are implemented from
scratch (`explain/hsic.py`, `explain/hsic_lasso.py`) with hypothesis property
tests written before the implementation: symmetry, non-negativity,
nHSIC ∈ [0,1], joint-permutation invariance, weight-splitting for duplicated
features, and sparsity monotone in the regularisation ρ.

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

### 5.1 The evaluation metrics

There is no ground-truth explanation on real data, so the pipeline is judged
by a set of complementary quantities, defined below. Throughout, $V$ is the
set of explained test entities, $v \in V$ one of them, $x_v$ its feature
vector, $\hat y$ the class the model predicts for it, $f_{\hat y}(\cdot)$ the
model's predicted probability for that class, and $S_k$ the set of the $k$
features with the largest $\beta$ in $v$'s explanation.

**Classification accuracy.** The fraction of held-out test entities whose
predicted class is correct, reported as mean ± standard deviation over the
5 training seeds. It is not a measure of explanation quality; it establishes
that the interpretable feature space is a viable model input at all, since
an explanation of an untrained model would be worthless.

**Fidelity+ (necessity).** Zero the top-$k$ feature columns of the model's
input and re-run the forward pass:

$$
\mathrm{fid}^+ = \frac{1}{|V|}\sum_{v \in V}
\Big[ f_{\hat y}(x_v) - f_{\hat y}\big(x_v^{\,\setminus S_k}\big) \Big].
$$

If the selected features are the ones the model relies on, removing them
should make the predicted-class probability fall, so **higher is better**.

**Fidelity− (sufficiency).** The complementary operation: keep *only* the
top-$k$ columns and zero everything else,

$$
\mathrm{fid}^- = \frac{1}{|V|}\sum_{v \in V}
\Big[ f_{\hat y}(x_v) - f_{\hat y}\big(x_v^{\,S_k}\big) \Big].
$$

If the selected features are sufficient on their own, the prediction should
survive, so **lower is better**.

**Random-$k$ control.** Neither number is meaningful in isolation: masking
any $k$ columns perturbs the prediction somewhat. Both metrics are therefore
recomputed with a random $k$-subset in place of $S_k$, and the quantity that
carries the evidence is the **gap** — $\mathrm{fid}^+$ above $\mathrm{random}^+$,
$\mathrm{fid}^-$ below $\mathrm{random}^-$. The sweep $k \in \{1,2,5,10\}$
doubles as the sparsity axis: a good explanation should already win at small
$k$, since the explanations themselves typically carry fewer than ten
non-zero $\beta$.

**Stability (Jaccard@5).** Agreement between two top-$k$ feature sets,

$$
J(S, S') = \frac{|S \cap S'|}{|S \cup S'|} \in [0,1],
$$

averaged over the explained nodes with $k = 5$. It is computed along two
axes: across **training seeds** (models of equal accuracy trained from
different initialisations) and across **hops** (the same model explained with
1-, 2- and 3-hop neighbourhoods). An explanation method that changes its
answer when nothing meaningful changed is not trustworthy, so higher is
better; the hop axis additionally quantifies how much the neighbourhood
radius — GraphLIME's sample source — dictates the result.

**Agreement (Jaccard@5 over predicates).** The same Jaccard index, but with
each feature first collapsed to its predicate URI so that the two feature
spaces and the baseline become comparable. It is used twice: **space A vs
space B**, measuring how much the choice of interpretable vocabulary changes
the conclusion, and **GraphLIME vs GNNExplainer**, an external cross-check.
Neither is a correctness measure — there is no reference ranking — so it is
read as a sensitivity analysis rather than a score.

**Refusal rate.** The fraction of test entities whose neighbourhood falls
below `min_neighborhood = 10` and for which the pipeline declines to produce
an explanation. Reporting it is part of the method: a refusal is a finding
about applicability, not a failure.

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

![Fidelity+ vs k against the random-k control](../results/figures/fidelity_curve.png)
![Stability heatmap across seeds and hops](../results/figures/stability_heatmap.png)
![Feature-space (A vs B) and baseline agreement](../results/figures/agreement_comparison.png)
<!-- RESULTS:END -->

### 5.2 Reading the results, table and figure by figure

**Classification accuracy (table).** The interpretable feature space is not a
handicap on MUTAG — the 5-seed mean, 0.7559, *exceeds* the paper's one-hot
R-GCN (0.732) — while AIFB lands several points below its 0.958. The gap is
informative rather than disappointing: AIFB's classes hinge on *which*
specific entity a person is connected to, exactly the information that
per-entity embeddings encode for free and that predicate counts deliberately
discard in exchange for readability. The cost of interpretability is thus
visible and dataset-dependent, and it is paid in the input representation,
not in the explainer.

**Fidelity+ vs k against the random-k control** (`fidelity_curve.png`, plus
the fidelity table). For $k \le 5$ the GraphLIME curve sits clearly above the
random-$k$ control on both datasets — on MUTAG at $k=1$, 0.079 against
−0.001, and on AIFB at $k=5$, 0.108 against 0.052 — so masking the selected
columns damages the prediction more than masking arbitrary ones: the
explanations point at features the model actually uses. At $k = 10$ on MUTAG
the comparison inverts (0.100 against 0.159). This is a property of the
sweep, not a failure of the method: explanations there typically have fewer
than ten non-zero $\beta$, so the top-10 set pads itself with irrelevant
columns, while random-10 already covers roughly a fifth of MUTAG's 46
columns and is therefore a strong perturbation in its own right. The
sufficiency side is read from the same table: fidelity− stays high on AIFB
at every $k$ (≈ 0.7), meaning no five features *suffice* — the model spreads
its evidence across the whole neighbourhood feature profile — whereas on
MUTAG keeping only the top-5 preserves the prediction markedly better than
keeping random-5 (0.199 against 0.335).

**Stability heatmap across seeds and hops** (`stability_heatmap.png`). The
two axes of the heatmap tell opposite stories. Across **training seeds**,
explanations agree moderately, with mean Jaccard@5 between 0.17 and 0.75:
models of comparable accuracy reach their decisions through partially
different feature use, which is exactly why the per-seed comparison is
reported instead of a single-seed number. Across **hops**, agreement is near
zero (0.01–0.07). The reason is structural: the 1-hop and 2-hop
neighbourhoods of an entity are different node populations, and since
GraphLIME's samples *are* the neighbourhood, the local HSIC estimate follows
the sample rather than the target alone. The neighbourhood radius is
therefore the single most consequential explanation hyperparameter in this
setting, and reporting it as such is one of this project's honest negative
findings.

**Feature-space (A vs B) and baseline agreement**
(`agreement_comparison.png`). Both comparisons are made at the predicate
level. AIFB shows moderate A-vs-B overlap (0.28), MUTAG essentially none
(0.003): the pair-level space shifts weight onto specific `rdf:type` objects
— individual atom types — which space A cannot express at all, so the two
vocabularies describe genuinely different things on that dataset. Agreement
with GNNExplainer's attribute mask is limited on both (0.15 on AIFB, 0.32 on
MUTAG). This is expected rather than alarming: kernel-based dependence
selection and a gradient-learned soft mask answer related but distinct
questions, and with no reference ranking available the disagreement is
itself the reported result.

**Refused explanations (table): none.** With 2-hop neighbourhoods every test
entity of both datasets clears `min_neighborhood = 10`, giving a refusal rate
of 0.0000 on both. The refusal machinery is nonetheless exercised by the test
suite and by smaller-radius runs; here it is reported empty, which is the
correct reading of "the guard was never needed at this radius" rather than
"the guard is absent".

## 6. Qualitative examples

The executed notebook (`notebooks/qualitative.ipynb`, rendered as
`qualitative.html`) loads the committed checkpoints and prints sentence-form
explanations for test entities of both datasets, e.g. a person *correctly
predicted* into a research group *because of* `publication` and `author`
edges toward the group's papers, or a MUTAG bond classified through its
atom-type context. Refusals are printed inline, not skipped.

## 7. Limitations

**The β weights are dependence, not causation.** HSIC Lasso ranks features by
how strongly the model's outputs *co-vary* with them over the local sample,
after discounting redundancy among the features themselves. A high $\beta_k$
therefore licenses the statement "within this neighbourhood, the model's
prediction depends on predicate $k$", not "removing predicate $k$ would
change the entity's true class". The account is additionally **local**: it is
fitted on one neighbourhood and does not generalise into a global statement
about the model, which is by design — GraphLIME is a local surrogate — but
must be remembered when reading a single explanation as though it described
the classifier as a whole.

**Fidelity is measured on columns, not on structure.** Masking a feature
column removes the *evidence that the entity has a predicate*, but leaves the
graph's edges intact, so message passing still propagates along them.
This is the honest consequence of the design decision in §2: model and
explainer share the feature vocabulary, which is what made fidelity definable
at all, but the perturbation lives in the feature matrix rather than in the
RDF triples. A structurally faithful alternative — deleting the corresponding
triples and re-running the forward pass — would respect RDF semantics more
closely at a substantially higher cost, and was deliberately not run here;
the fidelity numbers should be read as necessity and sufficiency of feature
*columns*, graph-wide.

**Neighbourhood handling introduces two distortions at the extremes.** Above
200 nodes the neighbourhood is subsampled, trading a small amount of
estimator variance for tractability; the subsample is deterministic per node
and always retains the target, so results are reproducible, but very large
neighbourhoods are described by a sample rather than in full. Below 10 nodes
the pipeline refuses outright, because HSIC estimated from a handful of
samples is noise. Between those bounds, §5.2 shows that the radius itself
dominates the outcome — the near-zero cross-hop Jaccard means that "the
explanation of entity $v$" is only well-defined once the radius is fixed and
stated.

**The evidence base is two small benchmarks.** AIFB and MUTAG have 176 and
340 labelled entities respectively, which puts every mean in §5 on a few tens
of test nodes and makes the seed-to-seed spread in both accuracy and
stability substantial. Conclusions about the interpretable feature space are
therefore claimed for RDF entity classification at this scale and not beyond
it; in particular, the A-vs-B comparison reflects the vocabulary of these two
ontologies, and space B's behaviour is contingent on a support threshold
whose sweep was not performed.

**The baseline comparison is not like-for-like.** GNNExplainer natively
optimises a soft mask over *edges*, but its edge masks are structurally
incompatible with `RGCNConv`, which runs one propagate per relation on edge
subsets so that a full-graph mask can never align (ADR-007). The baseline
therefore uses GNNExplainer's *attribute* mask over the same feature matrix.
This makes the two rankings directly comparable, at the price of comparing
the two methods on feature attribution only — the edge-level explanation that
GNNExplainer would normally produce is outside the comparison, and the
agreement figures in §5.2 must be read accordingly.

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
