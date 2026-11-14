"""Builds per-profile time-sorted outgoing-interaction sequences for the
Transformer encoder (Step 5).

Each token is [log1p(hours since previous event), target eventually replied
(any edge back, not a labeled "reply" — that label only exists in
edges_audit.csv), target was a newly-created account at contact time]. Capped
at `max_len` most recent events, zero-padded otherwise. Uses login/edit
timestamps-free event streams — only outgoing interaction edges, never the
behavioral timestamps M2-B already covers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from model5.graph_interaction.config import NEW_ACCOUNT_WINDOW_DAYS

TOKEN_DIM = 3


def build_sequences(
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    node_order: list[str],
    max_len: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (seq [N, max_len, 3] float32, pad_mask [N, max_len] bool —
    True means that position is padding, per torch's src_key_padding_mask
    convention)."""
    node_index = {pid: i for i, pid in enumerate(node_order)}
    created_lookup = profiles_df.set_index("profile_id")["created_at_dt"]

    out = edges_df[["source", "target", "timestamp_dt"]].copy()
    out = out[out["source"].isin(node_index)]

    pairs = edges_df[["source", "target"]].drop_duplicates()
    edge_set = set(zip(edges_df["source"], edges_df["target"]))
    pairs = pairs.copy()
    pairs["replied"] = [(t, s) in edge_set for s, t in zip(pairs["source"], pairs["target"])]
    out = out.merge(pairs, on=["source", "target"], how="left")

    out["target_created"] = out["target"].map(created_lookup)
    out["target_age_days"] = (out["timestamp_dt"] - out["target_created"]).dt.total_seconds() / 86400.0
    out["target_is_new"] = out["target_age_days"] <= NEW_ACCOUNT_WINDOW_DAYS

    out = out.sort_values(["source", "timestamp_dt"])
    out["rank_desc"] = out.groupby("source").cumcount(ascending=False)
    out = out[out["rank_desc"] < max_len]
    out = out.sort_values(["source", "timestamp_dt"])
    out["pos"] = out.groupby("source").cumcount()
    out["time_gap_hours"] = out.groupby("source")["timestamp_dt"].diff().dt.total_seconds().fillna(0.0) / 3600.0

    n = len(node_order)
    seq = np.zeros((n, max_len, TOKEN_DIM), dtype=np.float32)
    pad_mask = np.ones((n, max_len), dtype=bool)

    node_idx_arr = out["source"].map(node_index).to_numpy()
    pos_arr = out["pos"].to_numpy()

    seq[node_idx_arr, pos_arr, 0] = np.log1p(np.clip(out["time_gap_hours"].to_numpy(dtype=float), 0, None))
    seq[node_idx_arr, pos_arr, 1] = out["replied"].fillna(False).to_numpy(dtype=float)
    seq[node_idx_arr, pos_arr, 2] = out["target_is_new"].fillna(False).to_numpy(dtype=float)
    pad_mask[node_idx_arr, pos_arr] = False

    # A fully-masked row (no outgoing edges at all) makes a Transformer's
    # softmax attend over nothing -> NaN. Un-mask one zero-token position so
    # those nodes get a harmless, well-defined all-zero embedding instead.
    fully_padded = pad_mask.all(axis=1)
    pad_mask[fully_padded, 0] = False

    return torch.from_numpy(seq), torch.from_numpy(pad_mask)
