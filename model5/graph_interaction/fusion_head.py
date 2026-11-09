"""Fusion head (Step 5) + the combined Model 5 module.

Concatenates the GAT embedding (64) and, when enabled, the Transformer
sequence embedding (64) into up to 128 dims, plus a reserved zero-filled
slot for future cross-domain (e.g. social-media) features so that work
doesn't require redesigning this head. A 2-layer MLP maps the concatenation
to one logit; `torch.sigmoid` of that logit is graph_risk in [0, 1].
Returns logits (not sigmoid) from forward() — use BCEWithLogitsLoss for
training, and apply sigmoid only when reporting graph_risk.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model5.graph_interaction.gat_model import GATEncoder, HIDDEN_DIM
from model5.graph_interaction.sequence_model import D_MODEL, TransformerSequenceEncoder

CROSS_DOMAIN_RESERVED_DIM = 16


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logits, [N]


class GraphInteractionModel(nn.Module):
    """use_sequence=False -> GAT-only ablation. use_sequence=True -> GAT + Transformer.

    use_skip=True concatenates the raw (already-normalized) node features
    directly onto the GAT embedding before the classifier head — a residual/
    "jumping knowledge" style skip connection. Diagnosis: nodes whose fraud
    signal is pure self-magnitude (a star hub's huge degree, a financial_scam
    profile's narrow demographic-targeted outreach) sit next to mostly
    ordinary, mutually-unrelated legit neighbors, so 2 rounds of attention-
    weighted averaging dilutes that self-signal before it reaches the head.
    The skip gives the classifier an undiluted look at each node's own raw
    features alongside whatever relational signal the GAT extracted.
    """

    def __init__(self, in_dim: int, use_sequence: bool = True, use_skip: bool = True, use_gat: bool = True,
                 cross_domain_slot: int = CROSS_DOMAIN_RESERVED_DIM):
        super().__init__()
        self.use_gat = use_gat
        self.use_sequence = use_sequence
        # No GAT means no relational embedding at all, so the raw features
        # must be included regardless of use_skip, or the model sees nothing.
        self.use_skip = use_skip or not use_gat
        self.cross_domain_slot = cross_domain_slot
        fusion_in = cross_domain_slot
        if use_gat:
            self.gat = GATEncoder(in_dim)
            fusion_in += HIDDEN_DIM
        if use_sequence:
            self.seq_encoder = TransformerSequenceEncoder()
            fusion_in += D_MODEL
        if self.use_skip:
            fusion_in += in_dim
        self.head = FusionHead(fusion_in)

    def forward(self, x, edge_index, seq=None, seq_pad_mask=None) -> torch.Tensor:
        parts = []
        if self.use_gat:
            parts.append(self.gat(x, edge_index))
        if self.use_sequence:
            if seq is None or seq_pad_mask is None:
                raise ValueError("use_sequence=True requires seq and seq_pad_mask")
            parts.append(self.seq_encoder(seq, seq_pad_mask))
        if self.use_skip:
            parts.append(x)
        if self.cross_domain_slot:
            parts.append(torch.zeros(x.shape[0], self.cross_domain_slot, device=x.device, dtype=x.dtype))
        fused = torch.cat(parts, dim=1)
        return self.head(fused)
