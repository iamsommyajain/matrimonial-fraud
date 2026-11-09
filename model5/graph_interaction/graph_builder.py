"""Builds a torch_geometric Data object from edges.csv + node features.

Used later by the GAT (Steps 4-6). Built now so the data pipeline can be
sanity-checked ahead of time; nothing here trains a model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from model5.graph_interaction.features import RAW_FEATURE_COLUMNS


@dataclass
class GraphBuildResult:
    data: Data
    node_order: list[str]  # profile_id at each node index, index == position


def build_pyg_data(
    profiles_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    scaled_features_df: pd.DataFrame,
    train_ids: set,
    val_ids: set,
    test_ids: set,
) -> GraphBuildResult:
    node_order = profiles_df["profile_id"].tolist()
    node_index = {pid: i for i, pid in enumerate(node_order)}

    feat_lookup = scaled_features_df.set_index("profile_id")
    x = torch.tensor(feat_lookup.loc[node_order, RAW_FEATURE_COLUMNS].to_numpy(dtype=np.float32))

    valid_edges = edges_df[edges_df["source"].isin(node_index) & edges_df["target"].isin(node_index)]
    src = valid_edges["source"].map(node_index).to_numpy()
    dst = valid_edges["target"].map(node_index).to_numpy()
    edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)

    y = torch.tensor(profiles_df.set_index("profile_id").loc[node_order, "is_fraud"].astype(int).to_numpy(), dtype=torch.long)

    train_mask = torch.tensor([pid in train_ids for pid in node_order], dtype=torch.bool)
    val_mask = torch.tensor([pid in val_ids for pid in node_order], dtype=torch.bool)
    test_mask = torch.tensor([pid in test_ids for pid in node_order], dtype=torch.bool)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return GraphBuildResult(data=data, node_order=node_order)
