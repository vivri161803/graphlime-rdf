"""GNNExplainer baseline over the shared predicate vocabulary (plan M7, ADR-007).

PyG's edge-mask machinery is structurally incompatible with ``RGCNConv``
(the layer runs one propagate per relation on an edge *subset*, while the
explainer installs a full-graph mask — the sizes cannot match). Instead we
let GNNExplainer learn its **attribute mask** over the interpretable feature
matrix: it scores exactly the same named features GraphLIME scores, which
makes the two rankings directly comparable — and keeps the baseline honest
instead of silently broken.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.explain import Explainer, GNNExplainer

from graphlime_rdf.evaluate.agreement import predicate_of_feature
from graphlime_rdf.features.vocabulary import Vocabulary
from graphlime_rdf.models.rgcn import RGCN


def gnnexplainer_feature_scores(
    model: RGCN,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    node: int,
    epochs: int = 100,
    seed: int = 0,
) -> Tensor:
    """GNNExplainer attribute-mask importance per feature column (≥ 0)."""
    torch.manual_seed(seed)
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type=None,
        model_config={
            "mode": "multiclass_classification",
            "task_level": "node",
            "return_type": "raw",
        },
    )
    explanation = explainer(x, edge_index, index=node, edge_type=edge_type)
    node_mask: Tensor = explanation.node_mask.detach()
    return node_mask.sum(dim=0)


def gnnexplainer_predicate_ranking(
    model: RGCN,
    x: Tensor,
    edge_index: Tensor,
    edge_type: Tensor,
    vocabulary: Vocabulary,
    node: int,
    epochs: int = 100,
    seed: int = 0,
) -> list[tuple[str, float]]:
    """Rank predicates by summed attribute-mask weight — GraphLIME-comparable."""
    scores = gnnexplainer_feature_scores(
        model, x, edge_index, edge_type, node, epochs=epochs, seed=seed
    )
    totals: dict[str, float] = {}
    for j, name in enumerate(vocabulary.names):
        predicate = predicate_of_feature(name)
        totals[predicate] = totals.get(predicate, 0.0) + float(scores[j])
    return sorted(totals.items(), key=lambda item: (-item[1], item[0]))
