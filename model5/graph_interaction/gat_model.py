"""GAT structural encoder (Step 4).

2 GAT layers, 4 attention heads, hidden size 64, dropout 0.3. Attention lets
neighbors that look risky influence a node's embedding more than a plain
mean/sum aggregator would. Output: a 64-dim structural embedding per node.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

HIDDEN_DIM = 64
HEADS = 4
DROPOUT = 0.3


class GATEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = HIDDEN_DIM, heads: int = HEADS, dropout: float = DROPOUT):
        super().__init__()
        assert hidden_dim % heads == 0
        self.gat1 = GATConv(in_dim, hidden_dim // heads, heads=heads, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.gat1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gat2(h, edge_index)
        return h  # [N, hidden_dim]
