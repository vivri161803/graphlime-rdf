# ---
# jupytext:
#   formats: py:percent
#   text_representation:
#     extension: .py
#     format_name: percent
# ---

# %% [markdown]
# # GraphLIME on RDF — a short live demo
#
# Four steps, run live before the slides:
#
# 1. an RDF entity **has no features** — so there is nothing to explain;
# 2. we **build** the missing features out of the graph's own vocabulary;
# 3. on a graph whose answer we planted ourselves, the method **finds it**;
# 4. on the real datasets, the explanations **read as sentences**.
#
# Nothing is retrained on AIFB/MUTAG: the models come from the committed
# checkpoints. Rebuild this page with `just render` (~15 s).

# %%
from pathlib import Path

from graphlime_rdf.config import ExperimentConfig, SyntheticConfig
from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.data.synthetic import generate_ground_truth_graph
from graphlime_rdf.explain.graphlime import Explanation, explain_node
from graphlime_rdf.features import build_features
from graphlime_rdf.pipeline import test_nodes_of
from graphlime_rdf.training import load_checkpoint, train_run

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EXAMPLES_PER_DATASET = 4
TRIPLES_SHOWN = 5


def short(term: str) -> str:
    """Compact display form of a URI/term (literals keep their value)."""
    term = term.strip("<>")
    if term.startswith('"'):  # '"value"^^<datatype>' → '"value"'
        return term.split('"^^', 1)[0] + '"'
    for sep in ["#", "/"]:
        if sep in term:
            term = term.rsplit(sep, 1)[-1]
    return term or term


# %% [markdown]
# ## 1. An RDF entity has no features
#
# AIFB is a university's semantic portal — people, research groups,
# publications. The task: *which group does this person belong to?*

# %%
graph = load_rdf_graph("aifb", root=REPO / "data")
NODE = test_nodes_of(graph)[0]  # the first test entity, not a cherry-picked one
me = short(graph.node_names[NODE])

print(
    f"AIFB: {graph.num_nodes} entities, {graph.edge_index.shape[1]} triples, "
    f"{graph.num_relations} predicates, {graph.num_classes} classes\n"
    f"worked example: node {NODE} — {me}\n"
)

src, dst = graph.edge_index
outgoing = (src == NODE).nonzero().flatten()
incoming = (dst == NODE).nonzero().flatten()
for e in outgoing[:TRIPLES_SHOWN]:
    predicate = short(graph.relation_names[int(graph.edge_type[int(e)])])
    print(f"  ({me})  --{predicate}-->  {short(graph.node_names[int(dst[int(e)])])}")
for e in incoming[:TRIPLES_SHOWN]:
    predicate = short(graph.relation_names[int(graph.edge_type[int(e)])])
    print(f"  {short(graph.node_names[int(src[int(e)])])}  --{predicate}-->  ({me})")

print(f"\n{len(outgoing)} outgoing + {len(incoming)} incoming triples, and not one attribute.")

# %% [markdown]
# That is the whole difficulty. GraphLIME selects among the **columns of a
# node feature matrix** — in a citation network every column is a word. Here
# there is no matrix at all, and the usual R-GCN input is a one-hot entity id,
# which would only ever let an explainer say *"entity #5758 mattered"*.

# %% [markdown]
# ## 2. So we build the features from the graph
#
# **Space A** counts predicates around the node — `out:<p>`, `in:<p>`.
# **Space B** is finer, one column per predicate–object pair `out:<p>=<o>`.
# Either way every column is already a term of the ontology, so anything the
# selection returns is readable by construction.

# %%
config = ExperimentConfig.from_yaml(REPO / "configs" / "aifb.yaml")
x, vocabulary = build_features(graph, config.feature_space)
xb, vocab_b = build_features(
    graph, config.feature_space.model_copy(update={"kind": "predicate_object"})
)
print(f"space A (predicate counts):  {tuple(x.shape)}")
print(f"space B (predicate=object):  {tuple(xb.shape)}\n")

row = x[NODE]
print(f"the row of {me} in space A — {int((row > 0).sum())} non-zero of {x.shape[1]} columns:")
for j in row.nonzero().flatten():
    print(f"    {vocabulary.names[int(j)]:<55} {float(row[int(j)]):>6.0f}")

# %% [markdown]
# ## 3. Does it actually find the right predicate?
#
# On AIFB nobody knows the true reason for a label, so the claim is
# unfalsifiable there. We therefore build a graph in which one predicate *is*
# the class by construction, train on it, and check whether GraphLIME points
# at exactly that predicate. (This replays the project's hard gate,
# `tests/test_synthetic_gate.py`; without it no result is reported at all.)

# %%
syn_cfg = SyntheticConfig()
syn_graph = generate_ground_truth_graph(syn_cfg)
syn_config = ExperimentConfig.from_yaml(REPO / "configs" / "synthetic.yaml")
target = f"out:syn:p{syn_cfg.target_predicate}"
syn = train_run(syn_graph, syn_config, seed=0)

print(
    f"planted rule: class 1 ⟺ the entity has a {target} edge\n"
    f"({syn_cfg.num_entities} entities, {syn_cfg.num_predicates} predicates)\n"
    f"R-GCN test accuracy: {syn.manifest.final_test_accuracy:.4f}\n"
)

ranks = []
for node in test_nodes_of(syn_graph):
    out = explain_node(
        syn.model, node, syn.x, syn.edge_index, syn.edge_type,
        syn.vocabulary, syn_config.graphlime,
    )
    if isinstance(out, Explanation):
        names = [name for name, _ in out.top_features(len(out.beta))]
        ranks.append(names.index(target) + 1 if target in names else 10**6)

print(
    f"over {len(ranks)} test entities, {target} is ranked FIRST "
    f"{sum(r == 1 for r in ranks) / len(ranks):.1%} of the time."
)

# %% [markdown]
# ## 4. And on the real data, it speaks RDF
#
# Same machinery, real entities, read out as *"predicted group X because of
# predicate P toward Y"*. The model reads space A — its own vocabulary,
# checked by hash against the checkpoint — while GraphLIME regresses on the
# finer space B, so the explanation can name the term at the end of the edge.

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
# Each block is one test entity: the prediction with its confidence, the
# neighbourhood GraphLIME sampled, and the three predicates with the largest
# non-negative HSIC-Lasso weights β. The β are small in absolute terms because
# the weight is spread over a sparse solution across hundreds of (p,o)
# columns; what matters is which names come out on top. Refused nodes are
# printed as refusals, never hidden.
#
# ---
#
# That is the demo. How the method works, how faithful and how stable the
# explanations are, and where they break down — that is the slide deck.
