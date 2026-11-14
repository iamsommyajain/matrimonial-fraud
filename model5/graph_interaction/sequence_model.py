"""Transformer sequence encoder over per-profile outgoing-interaction
sequences (Step 5). 2 encoder layers, 4 heads, mean pooling over valid
(non-padded) positions. Output: a 64-dim sequence embedding per node.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model5.graph_interaction.sequence_builder import TOKEN_DIM

D_MODEL = 64
N_HEAD = 4
N_LAYERS = 2


class TransformerSequenceEncoder(nn.Module):
    def __init__(self, token_dim: int = TOKEN_DIM, d_model: int = D_MODEL, nhead: int = N_HEAD,
                 num_layers: int = N_LAYERS, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, d_model)
        # dim_feedforward defaults to 2048, sized for NLP-scale d_model (512+);
        # at our d_model=64 that's a ~32x-oversized FFN activation tensor per
        # layer (chunk_size x L x 2048) that alone was enough to exhaust 8GB
        # of RAM. 2x d_model is the usual ratio for a model this small.
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.d_model = d_model

    def forward(self, seq: torch.Tensor, pad_mask: torch.Tensor, chunk_size: int = 512) -> torch.Tensor:
        # seq: [N, L, token_dim]; pad_mask: [N, L] True == padded position.
        # Each node's sequence is independent of every other node's (no
        # cross-node attention here, unlike the GAT), so processing in
        # chunks is exactly equivalent to one big batch — it just bounds
        # peak memory. A single all-50k-nodes pass allocates a
        # [N, heads, L, L] attention tensor per layer (tens of GB); chunking
        # keeps that proportional to chunk_size instead of N.
        n = seq.shape[0]
        outputs = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            x = self.input_proj(seq[start:end])
            x = self.encoder(x, src_key_padding_mask=pad_mask[start:end])
            valid = (~pad_mask[start:end]).unsqueeze(-1).to(x.dtype)
            summed = (x * valid).sum(dim=1)
            counts = valid.sum(dim=1).clamp(min=1.0)
            outputs.append(summed / counts)
        return torch.cat(outputs, dim=0)  # [N, d_model]
