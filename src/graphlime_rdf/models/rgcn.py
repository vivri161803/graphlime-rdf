"""Two-layer R-GCN entity classifier (plan M3; Schlichtkrull et al., 2018).

Consumes the **interpretable predicate-indicator matrix** as input node
features `x` (the locked design decision, plan §6) instead of one-hot entity
ids, so the model and GraphLIME share one vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import RGCNConv

from graphlime_rdf.config import RGCNModelConfig


class RGCN(torch.nn.Module):
    """input features → RGCNConv → ReLU → dropout → RGCNConv → logits."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        num_relations: int,
        config: RGCNModelConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.conv1 = RGCNConv(
            in_dim, config.hidden_dim, num_relations, num_bases=config.num_bases
        )
        self.conv2 = RGCNConv(
            config.hidden_dim, num_classes, num_relations, num_bases=config.num_bases
        )

    def forward(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        h = F.relu(self.conv1(x, edge_index, edge_type))
        h = F.dropout(h, p=self.config.dropout, training=self.training)
        out: Tensor = self.conv2(h, edge_index, edge_type)
        return out

    @torch.no_grad()
    def predict_proba(self, x: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        """Softmax probabilities in eval mode — the signal GraphLIME explains."""
        self.eval()
        return torch.softmax(self.forward(x, edge_index, edge_type), dim=-1)
