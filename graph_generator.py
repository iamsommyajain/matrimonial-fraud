"""
graph_generator.py

Builds the interaction graph used by M5 (Graph Attention Network).

Nodes  = profiles (one node per profile_id)
Edges  = messages or match requests between profiles
        edge attributes: {message_count, match_requested, timestamp}

FRAUD TOPOLOGIES injected:
  Star      — hub contacts 150-400 victims directly
  Ring      — small group of fraud accounts cycle messages among themselves
  Relay     — A → B → C → D chain across fraud accounts
  Isolation — fraud node never receives replies (high out-degree, 0 in-degree)

The graph is serialized in two formats:
  - edges.csv       (profile_id_a, profile_id_b, weight, is_fraud_edge)
  - graph_stats.csv  (per-node degree stats, for quick inspection)

PyTorch Geometric will load from edges.csv in the M5 training script.
"""

import random
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta


def build_interaction_graph(profiles_df, fraud_type_col="fraud_type"):
    """
    Given the full profiles dataframe, build a directed interaction graph.

    Returns:
        G        : networkx DiGraph
        edges_df : DataFrame of edges with weights
    """
    G = nx.DiGraph()

    # Add all profiles as nodes with their fraud label
    for _, row in profiles_df.iterrows():
        G.add_node(
            row["profile_id"],
            is_fraud=row["is_fraud"],
            fraud_type=row.get("fraud_type", None),
        )

    legitimate_ids  = profiles_df[~profiles_df["is_fraud"]]["profile_id"].tolist()
    ring_profiles   = profiles_df[profiles_df["fraud_type"] == "coordinated_ring"]
    financial_frauds= profiles_df[profiles_df["fraud_type"] == "financial_scam"]
    other_frauds    = profiles_df[
        profiles_df["is_fraud"] &
        ~profiles_df["fraud_type"].isin(["coordinated_ring", "financial_scam"])
    ]

    edges = []

    # ── 1. Legitimate interaction edges ───────────────────────────────────
    # Legitimate users contact 3-15 others, get replies ~30-80% of the time.
    # Use a subset for efficiency (full pairwise is too slow for 47,500 nodes).
    legitimate_sample = legitimate_ids  # all of them

    for pid in legitimate_sample:
        row = profiles_df[profiles_df["profile_id"] == pid].iloc[0]
        n_contacts = min(int(row.get("unique_contacts", 5)), 20)
        targets = random.sample(
            [x for x in legitimate_ids if x != pid],
            k=min(n_contacts, len(legitimate_ids) - 1)
        )
        for tgt in targets:
            msg_count = random.randint(1, 15)
            edges.append({
                "source": pid, "target": tgt,
                "message_count": msg_count,
                "match_requested": random.random() < 0.3,
                "is_fraud_edge": False,
            })
            # Reciprocal reply ~50% of the time (bidirectional conversation)
            if random.random() < 0.5:
                edges.append({
                    "source": tgt, "target": pid,
                    "message_count": random.randint(1, msg_count),
                    "match_requested": False,
                    "is_fraud_edge": False,
                })

    # ── 2. Star topology — coordinated ring hubs ──────────────────────────
    hub_profiles = ring_profiles[ring_profiles["ring_role"] == "hub"] \
        if "ring_role" in ring_profiles.columns else ring_profiles.head(0)

    for _, hub_row in hub_profiles.iterrows():
        hub_id = hub_row["profile_id"]
        n_victims = random.randint(150, min(350, len(legitimate_ids)))
        victims = random.sample(legitimate_ids, k=n_victims)
        for v in victims:
            edges.append({
                "source": hub_id, "target": v,
                "message_count": random.randint(1, 5),
                "match_requested": True,
                "is_fraud_edge": True,
            })
            # Victims almost never reply
            if random.random() < 0.05:
                edges.append({
                    "source": v, "target": hub_id,
                    "message_count": 1,
                    "match_requested": False,
                    "is_fraud_edge": False,
                })

    # ── 3. Relay chain topology ───────────────────────────────────────────
    relay_profiles = ring_profiles[ring_profiles.get("ring_role", pd.Series()).eq("relay")] \
        if "ring_role" in ring_profiles.columns else ring_profiles.head(0)

    relay_ids = relay_profiles["profile_id"].tolist()
    if len(relay_ids) >= 3:
        # Build chains of length 3-5
        random.shuffle(relay_ids)
        i = 0
        while i + 2 < len(relay_ids):
            chain_len = min(random.randint(3, 5), len(relay_ids) - i)
            chain = relay_ids[i:i + chain_len]
            for j in range(len(chain) - 1):
                edges.append({
                    "source": chain[j], "target": chain[j + 1],
                    "message_count": random.randint(5, 20),
                    "match_requested": False,
                    "is_fraud_edge": True,
                })
            # Last in chain targets a legitimate victim
            if legitimate_ids:
                victim = random.choice(legitimate_ids)
                edges.append({
                    "source": chain[-1], "target": victim,
                    "message_count": random.randint(1, 5),
                    "match_requested": True,
                    "is_fraud_edge": True,
                })
            i += chain_len

    # ── 4. Financial scam outreach pattern ────────────────────────────────
    for _, frow in financial_frauds.iterrows():
        fid = frow["profile_id"]
        n_targets = int(frow.get("unique_contacts", 50))
        targets = random.sample(legitimate_ids, k=min(n_targets, len(legitimate_ids)))
        for tgt in targets:
            edges.append({
                "source": fid, "target": tgt,
                "message_count": random.randint(5, 25),
                "match_requested": True,
                "is_fraud_edge": True,
            })
            # Very rarely get replies
            if random.random() < 0.03:
                edges.append({
                    "source": tgt, "target": fid,
                    "message_count": random.randint(1, 3),
                    "match_requested": False,
                    "is_fraud_edge": False,
                })

    # ── 5. Other fraud profiles — sparse outreach ─────────────────────────
    for _, orow in other_frauds.iterrows():
        oid = orow["profile_id"]
        n_targets = int(orow.get("unique_contacts", 5))
        targets = random.sample(legitimate_ids, k=min(n_targets, len(legitimate_ids)))
        for tgt in targets:
            edges.append({
                "source": oid, "target": tgt,
                "message_count": random.randint(1, 10),
                "match_requested": random.random() < 0.4,
                "is_fraud_edge": True,
            })

    # Add all edges to graph
    for e in edges:
        G.add_edge(
            e["source"], e["target"],
            message_count=e["message_count"],
            match_requested=e["match_requested"],
            is_fraud_edge=e["is_fraud_edge"],
        )

    edges_df = pd.DataFrame(edges)

    # ── Per-node graph statistics (useful features for M5 node init) ──────
    stats = []
    for node in G.nodes():
        in_deg  = G.in_degree(node)
        out_deg = G.out_degree(node)
        in_w    = sum(d.get("message_count", 1) for _, _, d in G.in_edges(node, data=True))
        out_w   = sum(d.get("message_count", 1) for _, _, d in G.out_edges(node, data=True))
        stats.append({
            "profile_id":      node,
            "in_degree":       in_deg,
            "out_degree":      out_deg,
            "in_msg_weight":   in_w,
            "out_msg_weight":  out_w,
            "degree_ratio":    out_deg / max(in_deg, 1),   # high ratio = suspicious
            "response_rate":   in_deg  / max(out_deg, 1),  # low rate = suspicious
            "is_fraud":        G.nodes[node].get("is_fraud", False),
        })
    stats_df = pd.DataFrame(stats)

    return G, edges_df, stats_df