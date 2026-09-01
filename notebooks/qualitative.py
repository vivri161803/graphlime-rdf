# ---
# jupytext:
#   formats: py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # GraphLIME on RDF — live demo
#
# **This notebook is the opening act of the presentation.** It walks the whole
# argument of the project on a real knowledge graph, live, before a single
# slide is shown: the problem (an RDF entity has no feature vector), the fix
# (a feature space manufactured from the graph's own vocabulary), one
# explanation built end to end, the checks that keep it honest, and finally
# the recorded results that the slides then discuss.
#
# Nothing here is retrained and nothing is hard-coded: every number is
# recomputed from the committed checkpoints in `checkpoints/` and the raw RDF
# dumps in `data/`. The only cell that takes real time is the GNNExplainer
# baseline (§7, ~25 s); everything else is seconds.
#
# ```
# just render      # re-execute this notebook and rebuild qualitative.html
# ```
#
# **Demo route** — §1 the problem · §2 the feature space · §3 the model ·
# §4 one explanation · §5 fidelity · §6 stability · §7 baseline ·
# §8 the correctness gate · §9 qualitative gallery · §10 recorded results.

# %%
import time
from pathlib import Path

import numpy as np
import torch
from IPython.display import Image, Markdown, display

from graphlime_rdf.config import ExperimentConfig, SyntheticConfig
from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.evaluate.agreement import agreement_at_k, graphlime_predicate_ranking
from graphlime_rdf.evaluate.fidelity import fidelity_at_k
from graphlime_rdf.evaluate.stability import jaccard_at_k
from graphlime_rdf.explain.baseline import gnnexplainer_predicate_ranking
from graphlime_rdf.explain.graphlime import Explanation, explain_node
from graphlime_rdf.features import build_features
from graphlime_rdf.pipeline import AGREEMENT_K, FIDELITY_KS, STABILITY_K, test_nodes_of
from graphlime_rdf.training import load_checkpoint, train_run

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EXAMPLES_PER_DATASET = 4
TRIPLES_SHOWN = 6
HOPS_COMPARED = [1, 2, 3]


def short(term: str) -> str:
    """Compact display form of a URI/term (literals keep their value)."""
    term = term.strip("<>")
    if term.startswith('"'):  # '"value"^^<datatype>' → '"value"'
        return term.split('"^^', 1)[0] + '"'
    for sep in ["#", "/"]:
        if sep in term:
            term = term.rsplit(sep, 1)[-1]
    return term or term


def show_table(path: Path) -> None:
    """Render a generated markdown table from results/tables/."""
    display(Markdown(path.read_text()))


print(f"repository root: {REPO}")
print(f"torch {torch.__version__}")

# %% [markdown]
# ## 1. The problem, on the actual data
#
# AIFB is a slice of a university's semantic portal: people, research groups,
# publications, projects. The task is entity classification — *which research
# group does this person belong to?* The graph is loaded deterministically
# from the raw `.nt.gz` dump (node, relation and label indices are assigned in
# lexicographic order, so the mapping is identical in every process).

# %%
graph = load_rdf_graph("aifb", root=REPO / "data")
print(
    f"AIFB: {graph.num_nodes} entities, {graph.edge_index.shape[1]} triples, "
    f"{graph.num_relations} predicates, {graph.num_classes} classes\n"
    f"      {int(graph.train_mask.sum())} train / {int(graph.test_mask.sum())} test entities"
)

# The first test entity — not cherry-picked; it stays our worked example.
NODE = test_nodes_of(graph)[0]
print(f"\nworked example: node {NODE} — {short(graph.node_names[NODE])}")

# %% [markdown]
# What does the model actually get to see about this entity? In RDF, *nothing*
# but its edges — an entity carries no attributes of its own. Here are some of
# the triples it participates in:

# %%
src, dst = graph.edge_index
outgoing = (src == NODE).nonzero().flatten()
incoming = (dst == NODE).nonzero().flatten()
me = short(graph.node_names[NODE])

for e in outgoing[:TRIPLES_SHOWN]:
    predicate = short(graph.relation_names[int(graph.edge_type[int(e)])])
    print(f"  ({me})  --{predicate}-->  {short(graph.node_names[int(dst[int(e)])])}")
for e in incoming[:TRIPLES_SHOWN]:
    predicate = short(graph.relation_names[int(graph.edge_type[int(e)])])
    print(f"  {short(graph.node_names[int(src[int(e)])])}  --{predicate}-->  ({me})")
print(
    f"\n{len(outgoing)} outgoing + {len(incoming)} incoming triples — "
    f"and not one attribute. There is no feature vector to explain."
)

# %% [markdown]
# This is the crux of the project. GraphLIME's whole mechanism is a selection
# over *feature dimensions*: it asks which columns of a node feature matrix
# the model's output depends on. In a citation network every column is a word,
# readable by construction. Here there is no matrix at all — and the usual
# R-GCN input, a one-hot entity id, would only ever let an explainer say
# *"entity #5758 mattered"*. The explainer is not weakened; its input is
# undefined.

# %% [markdown]
# ## 2. Manufacturing the feature space
#
# We build the missing matrix from the graph itself, in the graph's own
# vocabulary. **Space A** counts predicates: `out:<p>` how many `<p>` edges
# leave the node, `in:<p>` how many enter it. **Space B** is finer, one column
# per predicate–object pair `out:<p>=<o>`. Both are named, so every column a
# selection can return is already a term of the ontology.

# %%
config = ExperimentConfig.from_yaml(REPO / "configs" / "aifb.yaml")
x, vocabulary = build_features(graph, config.feature_space)
space_b = config.feature_space.model_copy(update={"kind": "predicate_object"})
xb, vocab_b = build_features(graph, space_b)

print(f"space A (predicate counts):     {tuple(x.shape)}   vocabulary hash {vocabulary.hash}")
print(f"space B (predicate=object):     {tuple(xb.shape)}   vocabulary hash {vocab_b.hash}\n")

row = x[NODE]
print(f"the row of {me} in space A — {int((row > 0).sum())} non-zero of {x.shape[1]} columns:")
for j in row.nonzero().flatten():
    print(f"    {vocabulary.names[int(j)]:<55} {float(row[int(j)]):>6.0f}")

# %% [markdown]
# ## 3. The model
#
# An R-GCN trained on space A, loaded from the committed checkpoint of the
# best seed. The bundle is self-contained — weights, resolved config, label
# map and the vocabulary itself — so the notebook never retrains anything.

# %%
ckpt = load_checkpoint(REPO / "checkpoints" / "aifb_best.pt")
manifest = ckpt.manifest
print(
    f"checkpoint  seed={manifest.seed}  test accuracy={manifest.test_accuracy:.4f}\n"
    f"            trained at commit {manifest.git_commit} on {manifest.created_at[:10]}\n"
    f"            hidden_dim={manifest.resolved_config.model.hidden_dim}, "
    f"num_bases={manifest.resolved_config.model.num_bases}, "
    f"epochs={manifest.resolved_config.training.epochs}"
)

# The decisive invariant: the matrix we just built *is* the model's input.
assert vocabulary.hash == manifest.vocabulary_hash
print(
    f"\nvocabulary hash {vocabulary.hash} == checkpoint's {manifest.vocabulary_hash}\n"
    "→ the features the model consumes and the names the explanation returns\n"
    "  are the same objects. No translation layer, nothing to misalign."
)

edge_index, edge_type = graph.doubled_edges()
probs = ckpt.model.predict_proba(x, edge_index, edge_type)
label_names = {i: name for name, i in ckpt.label_map.items()}
predicted = int(probs[NODE].argmax())
print(
    f"\nprediction for {me}: {short(label_names[predicted])} "
    f"(p={float(probs[NODE, predicted]):.2f}), "
    f"true label {short(graph.label_names[int(graph.labels[NODE])])}"
)

# %% [markdown]
# ## 4. One explanation, end to end
#
# GraphLIME samples the node's k-hop neighbourhood, takes those rows of the
# feature matrix and the model's output probabilities on the same rows, and
# runs a non-negative HSIC Lasso: which feature columns carry the local
# dependence between input and prediction? The weights β are non-negative and
# sparse, and each one is attached to a predicate name.

# %%
started = time.perf_counter()
explanation = explain_node(
    ckpt.model, NODE, x, edge_index, edge_type, vocabulary, config.graphlime
)
elapsed = time.perf_counter() - started
assert isinstance(explanation, Explanation)

print(
    f"neighbourhood: {explanation.neighborhood_size} nodes at "
    f"hops={config.graphlime.hops}, ρ={config.graphlime.rho}   "
    f"[{elapsed:.2f} s]\n"
    f"sparsity: {explanation.sparsity:.3f} of {len(vocabulary)} columns are exactly zero\n"
)
selected = explanation.top_features(len(explanation.beta))
total = sum(beta for _, beta in selected)
print(f"{len(selected)} of {len(vocabulary)} columns received a non-zero weight:\n")
for name, beta in selected[:5]:
    direction, predicate = name.split(":", 1)
    word = "outgoing" if direction == "out" else "incoming"
    print(f"    β={beta:.4f}  ({100 * beta / total:>4.1f}% of the weight)  "
          f"{word:>8}  {short(predicate)}")

print(
    f"\n→ {me} is predicted to belong to {short(label_names[predicted])}\n"
    f"  because of its {short(explanation.top_features(1)[0][0])} edges."
)

# %% [markdown]
# ## 5. Is the explanation faithful? — fidelity±
#
# A readable explanation is not necessarily a true one. We zero the top-k
# columns graph-wide and watch the predicted probability: **fidelity+** =
# p(full) − p(top-k removed), high is good — the model really needed them.
# **fidelity−** = p(full) − p(only top-k kept), low is good — they suffice on
# their own. Against each we run a random-k control on the same node.

# %%
ranked_columns = [int(j) for j in np.argsort(-explanation.beta, kind="stable")]
print(f"{'k':>3}  {'fidelity+':>10} {'random+':>9}   {'fidelity−':>10} {'random−':>9}")
for k in FIDELITY_KS:
    fid_plus, fid_minus, rand_plus, rand_minus = fidelity_at_k(
        ckpt.model, x, edge_index, edge_type, NODE, ranked_columns, k
    )
    print(
        f"{k:>3}  {fid_plus:>10.4f} {rand_plus:>9.4f}   "
        f"{fid_minus:>10.4f} {rand_minus:>9.4f}"
    )
print("\n(one node only — the aggregate over all test nodes is in §10)")

# %% [markdown]
# Read this honestly. The prediction for this entity is saturated at p = 1.00,
# so removing a handful of columns does not move it at all and fidelity+ is
# flat — a single confident node simply cannot separate a good explanation
# from a bad one. What the fidelity− column does say is that the selected
# features alone recover the prediction better than a random set of the same
# size. The verdict belongs to the average over all test nodes and all values
# of k, in §10; that is why the reported metric is never read off one example.

# %% [markdown]
# ## 6. Is it stable? — the same node at 1, 2 and 3 hops
#
# The neighbourhood radius is a choice, not a fact about the data. If the
# explanation survives changing it, the selection is about the entity; if it
# does not, we should say so. Jaccard@5 over the top-5 feature sets:

# %%
by_hops = {}
for hops in HOPS_COMPARED:
    gl = config.graphlime.model_copy(update={"hops": hops})
    out = explain_node(ckpt.model, NODE, x, edge_index, edge_type, vocabulary, gl)
    assert isinstance(out, Explanation)
    by_hops[hops] = out
    top = ", ".join(short(name.split(":", 1)[1]) for name, _ in out.top_features(3))
    print(f"hops={hops}  ({out.neighborhood_size:>4} nodes)  top-3: {top}")

print()
for a, b in zip(HOPS_COMPARED, HOPS_COMPARED[1:], strict=False):
    print(
        f"Jaccard@{STABILITY_K}  hops={a} vs hops={b}: "
        f"{jaccard_at_k(by_hops[a], by_hops[b], STABILITY_K):.4f}"
    )

# %% [markdown]
# ## 7. Against a baseline — GNNExplainer on the same features
#
# PyG's edge-mask machinery is structurally incompatible with `RGCNConv`, so
# the baseline learns its **attribute mask** over the very same named feature
# matrix. Both explainers therefore rank the same predicates and the
# comparison is honest. This is the slow cell of the demo (~25 s: the baseline
# optimises a mask for 100 epochs, while GraphLIME above closed a convex
# problem in a fraction of a second).

# %%
started = time.perf_counter()
baseline_ranking = gnnexplainer_predicate_ranking(
    ckpt.model, x, edge_index, edge_type, vocabulary, NODE
)
print(f"GNNExplainer: {time.perf_counter() - started:.1f} s\n")

graphlime_ranking = graphlime_predicate_ranking(explanation)
print(f"{'GraphLIME':<20} {'β':>9}   {'GNNExplainer':<20} {'mask':>9}")
for rank in range(AGREEMENT_K):
    ga, wa = graphlime_ranking[rank] if rank < len(graphlime_ranking) else ("—", 0.0)
    gb, wb = baseline_ranking[rank] if rank < len(baseline_ranking) else ("—", 0.0)
    print(f"{short(ga):<20} {wa:>9.4f}   {short(gb):<20} {wb:>9.4f}")
print(
    f"\nagreement@{AGREEMENT_K} (Jaccard of the top-{AGREEMENT_K} predicate sets): "
    f"{agreement_at_k(graphlime_ranking, baseline_ranking, AGREEMENT_K):.4f}"
)

# %% [markdown]
# Look at the baseline's weights, not just its names: its attribute mask
# leaves most predicates tied at zero on this node, so its "top-5" is
# whatever the alphabet puts first and the agreement collapses to 0. The two
# explainers are not measuring the same thing — GraphLIME solves a convex
# selection over the neighbourhood's feature rows, GNNExplainer optimises a
# per-run mask — and the aggregate agreement (§10) is low on AIFB and moderate
# on MUTAG. A baseline that disagrees is reported as such, not tuned until it
# agrees.

# %% [markdown]
# ## 8. The correctness gate — a graph whose answer we already know
#
# Everything above is unverifiable on real data: nobody knows the true reason
# for an AIFB label. So the project's hard gate runs on a synthetic RDF graph
# in which one predicate *is* the class by construction. If GraphLIME cannot
# recover a planted answer, no number on AIFB or MUTAG may be reported.
#
# The gate itself lives in `tests/test_synthetic_gate.py` (`just synthetic`);
# what follows replays it live, over all test entities of the graph.

# %%
syn_cfg = SyntheticConfig()
syn_graph = generate_ground_truth_graph(syn_cfg)
syn_config = ExperimentConfig.from_yaml(REPO / "configs" / "synthetic.yaml")
target_feature = f"out:syn:p{syn_cfg.target_predicate}"

started = time.perf_counter()
syn_result = train_run(syn_graph, syn_config, seed=0)
print(
    f"planted rule: class 1 ⟺ the entity has a {target_feature} edge "
    f"({syn_cfg.num_entities} entities, {syn_cfg.num_predicates} predicates)\n"
    f"R-GCN test accuracy: {syn_result.manifest.final_test_accuracy:.4f} "
    f"[trained in {time.perf_counter() - started:.1f} s]\n"
)

ranks = []
for node in test_nodes_of(syn_graph):
    out = explain_node(
        syn_result.model, node, syn_result.x, syn_result.edge_index,
        syn_result.edge_type, syn_result.vocabulary, syn_config.graphlime,
    )
    if isinstance(out, Explanation):
        names = [name for name, _ in out.top_features(len(out.beta))]
        ranks.append(names.index(target_feature) + 1 if target_feature in names else 10**6)

top1 = sum(r == 1 for r in ranks) / len(ranks)
print(
    f"over {len(ranks)} test entities, {target_feature} is ranked FIRST "
    f"{top1:.1%} of the time\n"
    f"(the gate's contractual threshold is 95%, and it is never lowered)"
)

# %% [markdown]
# ## 9. Qualitative gallery — explanations as sentences
#
# The same machinery over several test entities of both datasets, read out as
# *"predicted group X because of predicate P toward Y"*. Here we switch the
# interpretable matrix to **space B** so the explanation can name the specific
# term at the end of the edge; the model input stays space A (its own
# vocabulary, verified by hash above). That is exactly the feature-space
# flexibility measured in the agreement experiment.

# %%
def explain_examples(dataset: str) -> None:
    """Sentence-form explanations in feature space B: predicate *toward* term."""
    ckpt = load_checkpoint(REPO / "checkpoints" / f"{dataset}_best.pt")
    cfg = ckpt.manifest.resolved_config
    graph = load_rdf_graph(dataset, root=REPO / "data")
    x, vocabulary = build_features(graph, cfg.feature_space)
    assert vocabulary.hash == ckpt.manifest.vocabulary_hash
    space_b = cfg.feature_space.model_copy(update={"kind": "predicate_object"})
    xb, vocab_b = build_features(graph, space_b)
    # Qualitative display: gentler regularisation than the quantitative runs
    # (rho=0.1 in results/) so the sparser (p,o) features surface; stated here
    # openly — the recorded metrics never use this value.
    gl_cfg = cfg.graphlime.model_copy(update={"rho": 0.01})
    edge_index, edge_type = graph.doubled_edges()
    probs = ckpt.model.predict_proba(x, edge_index, edge_type)
    label_names = {i: name for name, i in ckpt.label_map.items()}

    shown = 0
    for node in test_nodes_of(graph):
        out = explain_node(
            ckpt.model, node, x, edge_index, edge_type, vocab_b, gl_cfg,
            interpretable_x=xb,
        )
        if not isinstance(out, Explanation):
            print(f"— node {node}: refused ({out.reason})\n")
            continue
        predicted = int(probs[node].argmax())
        correct = "correctly" if predicted == int(graph.labels[node]) else "INCORRECTLY"
        top = out.top_features(3)
        reasons = "; ".join(
            f"{short(name.split(':', 1)[1].split('=')[0])} "
            f"({'outgoing' if name.startswith('out:') else 'incoming'}, β={beta:.2f})"
            + (f" toward {short(name.split('=', 1)[1])}" if "=" in name else "")
            for name, beta in top
        ) or f"no feature selected at ρ={gl_cfg.rho} (locally uniform prediction)"
        print(
            f"{graph.dataset.upper()} node {node} — {short(graph.node_names[node])}\n"
            f"  {correct} predicted {short(label_names[predicted])} "
            f"(p={float(probs[node, predicted]):.2f}, "
            f"neighborhood={out.neighborhood_size})\n"
            f"  because of: {reasons}\n"
        )
        shown += 1
        if shown >= EXAMPLES_PER_DATASET:
            break


# %% [markdown]
# ### AIFB — which research group does a person belong to?

# %%
explain_examples("aifb")

# %% [markdown]
# ### MUTAG — is a compound mutagenic?

# %%
explain_examples("mutag")

# %% [markdown]
# ### Reading the output
#
# Each block is one test entity: the model's prediction with its confidence,
# the neighbourhood size GraphLIME sampled, and the three predicates with the
# largest non-negative HSIC-Lasso weights β. A comparatively large β
# concentrated on one predicate means the local dependence between features
# and prediction runs mostly through that predicate; absolute magnitudes are
# small because β is shared across a sparse solution over hundreds of
# (p,o)-pair features. Refused nodes (neighbourhood below
# `min_neighborhood`) are printed as refusals — they are counted in
# `results/tables/refusals.md`, not hidden.

# %% [markdown]
# ## 10. The recorded results — and the handover to the slides
#
# Everything above was one node, or one graph, computed live. The tables below
# are the canonical numbers: generated by `just tables` from the validated
# JSONL records in `results/`, over all test entities and all five seeds.
# Every figure quoted in the report and in the slide deck comes from here.

# %%
TABLES = REPO / "results" / "tables"
for title, name in [
    ("### Classification accuracy — 5 seeds", "main_results.md"),
    ("### Fidelity± against the random-k control", "fidelity.md"),
    ("### Stability across seeds and hops (Jaccard@5)", "stability.md"),
    ("### Feature-space sensitivity and baseline agreement", "agreement.md"),
    ("### Refusals", "refusals.md"),
]:
    display(Markdown(title))
    show_table(TABLES / name)

# %%
FIGURES = REPO / "results" / "figures"
for name in ["fidelity_curve.png", "stability_heatmap.png", "agreement_comparison.png"]:
    display(Image(filename=str(FIGURES / name)))

# %% [markdown]
# ---
#
# **Where this leaves the presentation.** The demo has shown that the pipeline
# runs, that the explanations are in the vocabulary of the graph, and that the
# method recovers a planted ground truth. What it cannot show in one node is
# whether the selections beat their controls at scale, how far the two feature
# spaces diverge, or where the method breaks down — that is what the tables
# above summarise and what the slides now discuss, from
# `report/presentazione.pdf` (full argument in `report/relazione.pdf`).
