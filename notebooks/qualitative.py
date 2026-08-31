# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Qualitative explanations — GraphLIME on AIFB and MUTAG
#
# Worked examples (plan M8): for a handful of test entities per dataset we
# load the committed best checkpoint (no retraining), run GraphLIME, and read
# the explanation as a sentence — *"predicted group X because of predicate P
# toward Y"*. Everything shown here is recomputed live from
# `checkpoints/<dataset>_best.pt`.

# %%
from pathlib import Path

from graphlime_rdf.data.loader import load_rdf_graph
from graphlime_rdf.explain.graphlime import Explanation, explain_node
from graphlime_rdf.features import build_features
from graphlime_rdf.training import load_checkpoint

REPO = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
EXAMPLES_PER_DATASET = 4


def short(term: str) -> str:
    """Compact display form of a URI/term."""
    term = term.strip("<>")
    for sep in ["#", "/"]:
        if sep in term:
            term = term.rsplit(sep, 1)[-1]
    return term or term


def explain_examples(dataset: str) -> None:
    """Sentence-form explanations in feature space B: predicate *toward* term.

    The model input stays space A (its own vocabulary, verified by hash);
    GraphLIME regresses on the richer (p,o)-pair features, which is exactly
    the feature-space flexibility measured in the agreement experiment.
    """
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
    for node in sorted(int(i) for i in graph.test_mask.nonzero().flatten()):
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
# ## AIFB — which research group does a person belong to?
#
# The classifier sees only predicate-count features; GraphLIME tells us which
# predicates around the person carried the decision.

# %%
explain_examples("aifb")

# %% [markdown]
# ## MUTAG — is a compound mutagenic?

# %%
explain_examples("mutag")

# %% [markdown]
# ## Reading the output
#
# Each block is one test entity: the model's prediction with its confidence,
# the neighbourhood size GraphLIME sampled, and the three predicates with the
# largest non-negative HSIC-Lasso weights β. A β of 0.30+ concentrated on one
# predicate means the local dependence between features and prediction runs
# almost entirely through that predicate. Refused nodes (neighbourhood below
# `min_neighborhood`) are printed as refusals — they are counted in
# `results/tables/refusals.md`, not hidden.
