"""
generate_dataset.py

Orchestrates the full 50,000-profile synthetic dataset generation.

Run with:
    python generate_dataset.py --seed 42 --output_dir ./output

Output files:
    output/profiles.csv        — flat profile table (M1, M2, M3, M4 features)
    output/profiles.parquet    — same, faster format for ML training
    output/edges.csv           — interaction graph edges (M5)
    output/graph_stats.csv     — per-node degree statistics (M5 node features)
    output/splits/train.csv    — temporal split: months 1-4
    output/splits/val.csv      —                 month 5
    output/splits/test.csv     —                 month 6
    output/dataset_report.txt  — class balance and signal statistics

Dataset composition (50,000 profiles):
    47,500  legitimate         (95%)
       500  functional fraud   (1%)
       500  template bio fraud (1%)
       500  image fraud        (1%)
       500  financial scam     (1%)
       500  coordinated ring   (hub + satellite, 0.75%)
       500  multi-signal fraud (1%)  — 3,000 fraud total = 6% for harder eval
    ──────
    50,000
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add dataset/ to path so imports work whether run from root or dataset/
sys.path.insert(0, os.path.dirname(__file__))

from legitimate_generator import generate_legitimate_profile
from fraud_injector import (
    inject_functional_fraud,
    inject_template_bio_fraud,
    inject_financial_scam_signals,
    inject_ring_signals,
    inject_multi_signal_fraud,
)
from graph_generator import build_interaction_graph
from report_generator import generate_all_reports


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

N_LEGITIMATE   = 47_500
N_FUNCTIONAL   =    500
N_TEMPLATE_BIO =    500
N_IMAGE        =    500
N_FINANCIAL    =    500
N_RING         =    500   # composed of: 10 hubs + 50 relays + 440 satellites
N_MULTI        =    500

# Ring breakdown
N_RING_HUBS      = 10
N_RING_RELAYS    = 50
N_RING_SATELLITES= N_RING - N_RING_HUBS - N_RING_RELAYS  # 440

# Simulation period: 6 months for temporal split
SIM_START = datetime(2024, 1, 1)
SIM_END   = datetime(2024, 6, 30)

# Template indices — fraud profiles within the same "campaign" share a template
N_BIO_CAMPAIGNS = 10   # 10 campaigns, 50 profiles each use same template


# ─────────────────────────────────────────────────────────────────────────────
# Timestamp sampler
# ─────────────────────────────────────────────────────────────────────────────

def random_created_at(month_range=(1, 6)):
    """Sample a creation timestamp within [month_range] months of SIM_START."""
    start_day = (month_range[0] - 1) * 30
    end_day   = month_range[1] * 30
    day_offset = random.randint(start_day, end_day)
    hour       = random.randint(0, 23)
    minute     = random.randint(0, 59)
    return SIM_START + timedelta(days=day_offset, hours=hour, minutes=minute)


def ring_created_at(base_time, jitter_hours=6):
    """Ring members are created in a tight time window — key graph signal."""
    offset = timedelta(hours=random.uniform(0, jitter_hours))
    return base_time + offset


# ─────────────────────────────────────────────────────────────────────────────
# Generation functions
# ─────────────────────────────────────────────────────────────────────────────

def generate_legitimate_batch(n, desc="Generating legitimate profiles"):
    profiles = []
    for _ in tqdm(range(n), desc=desc, ncols=80):
        ts = random_created_at((1, 6))
        profiles.append(generate_legitimate_profile(created_at=ts))
    return profiles


def generate_functional_fraud_batch(n):
    profiles = []
    for _ in tqdm(range(n), desc="Injecting functional fraud", ncols=80):
        ts   = random_created_at((1, 6))
        base = generate_legitimate_profile(created_at=ts)
        profiles.append(inject_functional_fraud(base))
    return profiles


def generate_template_bio_batch(n):
    """
    Divide into campaigns. Each campaign uses the same template index so
    M2's cosine similarity across profiles within a campaign is high.
    """
    profiles = []
    campaign_size = n // N_BIO_CAMPAIGNS
    for campaign_idx in range(N_BIO_CAMPAIGNS):
        template_idx = campaign_idx  # campaigns 0-9 → templates 0-9
        for _ in tqdm(
            range(campaign_size),
            desc=f"Template bio campaign {campaign_idx + 1}/{N_BIO_CAMPAIGNS}",
            ncols=80
        ):
            ts   = random_created_at((1, 6))
            base = generate_legitimate_profile(created_at=ts)
            profiles.append(inject_template_bio_fraud(base, shared_template_idx=template_idx))
    # Fill remainder
    remainder = n - len(profiles)
    for _ in range(remainder):
        ts   = random_created_at((1, 6))
        base = generate_legitimate_profile(created_at=ts)
        profiles.append(inject_template_bio_fraud(base))
    return profiles



def generate_financial_scam_batch(n):
    profiles = []
    for _ in tqdm(range(n), desc="Injecting financial scam", ncols=80):
        ts   = random_created_at((1, 6))
        base = generate_legitimate_profile(created_at=ts)
        profiles.append(inject_financial_scam_signals(base))
    return profiles


def generate_ring_batch(n_hubs, n_relays, n_satellites):
    """
    Generate coordinated ring profiles.
    Rings are created in batches with tight timestamp clustering —
    this is the key graph signal M5 should learn.

    We create 5 distinct rings, each with:
        2 hubs + 10 relays + ~88 satellites
    """
    profiles = []
    n_rings = 5
    hubs_per_ring      = n_hubs // n_rings
    relays_per_ring    = n_relays // n_rings
    satellites_per_ring= n_satellites // n_rings

    for ring_idx in range(n_rings):
        ring_id = f"ring_{ring_idx:02d}"
        # All members of a ring cluster around a single creation timestamp
        ring_creation_base = random_created_at((1, 5))  # rings don't appear in month 6

        # Hubs
        for _ in range(hubs_per_ring):
            ts   = ring_created_at(ring_creation_base, jitter_hours=2)
            base = generate_legitimate_profile(created_at=ts)
            profiles.append(inject_ring_signals(base, ring_id=ring_id, role="hub"))

        # Relays
        for _ in range(relays_per_ring):
            ts   = ring_created_at(ring_creation_base, jitter_hours=4)
            base = generate_legitimate_profile(created_at=ts)
            profiles.append(inject_ring_signals(base, ring_id=ring_id, role="relay"))

        # Satellites
        for _ in tqdm(
            range(satellites_per_ring),
            desc=f"Ring {ring_idx + 1}/{n_rings} satellites",
            ncols=80
        ):
            ts   = ring_created_at(ring_creation_base, jitter_hours=6)
            base = generate_legitimate_profile(created_at=ts)
            profiles.append(inject_ring_signals(base, ring_id=ring_id, role="satellite"))

    return profiles


def generate_multi_signal_batch(n):
    profiles = []
    for i in tqdm(range(n), desc="Multi-signal fraud", ncols=80):
        ts   = random_created_at((1, 6))
        base = generate_legitimate_profile(created_at=ts)
        # Rotate template indices so multi profiles also contribute to campaign clusters
        tmpl_idx = i % N_BIO_CAMPAIGNS
        profiles.append(inject_multi_signal_fraud(base, shared_template_idx=tmpl_idx))
    return profiles


def generate_and_save_reports(all_profiles, output_dir):
    print("\nGenerating community reports (Model 4 data)...")
    all_reports = generate_all_reports(all_profiles)
    reports_df  = pd.DataFrame(all_reports)
 
    reports_path = os.path.join(output_dir, "reports.csv")
    reports_df.to_csv(reports_path, index=False)
 
    print(f"  Reports generated: {len(reports_df):,}")
    print(f"  Fraud reports:     {reports_df['reported_is_fraud'].sum():,}")
    print(f"  Noise reports:     {(~reports_df['reported_is_fraud']).sum():,}")
    print(f"Saved: {reports_path}")
    return reports_df


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split(df):
    """
    Split by creation month. Simulates real deployment:
        Train: months 1-4 (Jan-Apr 2024)
        Val:   month 5   (May 2024)
        Test:  month 6   (Jun 2024)

    This tests generalisation to FUTURE fraud patterns, not random shuffled
    test sets that leak temporal information.
    """
    df["created_month"] = pd.to_datetime(
    df["created_at"],
    format="ISO8601",
    errors="coerce"
).dt.month
    train = df[df["created_month"] <= 4].drop(columns=["created_month"])
    val   = df[df["created_month"] == 5].drop(columns=["created_month"])
    test  = df[df["created_month"] == 6].drop(columns=["created_month"])
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def flatten_for_csv(profiles):
    """
    The profile dicts contain nested structures (lists, dicts) that can't go
    directly into a flat CSV. This function:
      - Serialises lists as JSON strings (login_timestamps…)
      - Flattens injected_signals → pipe-separated string
      - Drops internal-only fields (_template_idx)
    """
    flat = []
    for p in profiles:
        row = {}
        for k, v in p.items():
            if k.startswith("_"):           # internal metadata
                continue
            if isinstance(v, list):
                row[k] = json.dumps(v)      # JSON string for complex lists
            elif isinstance(v, dict):
                row[k] = json.dumps(v)
            else:
                row[k] = v
        # Injected signals as pipe-separated for easy filtering
        row["injected_signals_str"] = "|".join(p.get("injected_signals", []))
        flat.append(row)
    return flat


def write_dataset_report(df, output_dir):
    """Write a human-readable summary of the dataset composition."""
    lines = ["=" * 60, "MATRIMONIAL FRAUD DATASET — GENERATION REPORT", "=" * 60]

    lines.append(f"\nTotal profiles:  {len(df):,}")
    lines.append(f"Fraud profiles:  {df['is_fraud'].sum():,}")
    lines.append(f"Fraud rate:      {df['is_fraud'].mean() * 100:.1f}%\n")

    lines.append("Class breakdown:")
    vc = df["fraud_type"].fillna("legitimate").value_counts()
    for label, count in vc.items():
        pct = count / len(df) * 100
        lines.append(f"  {label:<25} {count:>6,}   ({pct:.1f}%)")

    lines.append("\nFraud severity:")
    sv = df["fraud_severity"].fillna("none").value_counts()
    for s, c in sv.items():
        lines.append(f"  {s:<10} {c:>6,}")

    lines.append("\nTop injected signals:")
    all_signals = []
    for s in df["injected_signals_str"].dropna():
        all_signals.extend(s.split("|"))
    from collections import Counter
    for sig, cnt in Counter(all_signals).most_common(15):
        if sig:
            lines.append(f"  {sig:<40} {cnt:>5,}")

    report = "\n".join(lines)
    with open(os.path.join(output_dir, "dataset_report.txt"), "w") as f:
        f.write(report)
    print("\n" + report)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(seed=42, output_dir="./output", skip_graph=False):
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "splits"), exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Matrimonial Fraud Dataset Generator  |  seed={seed}")
    print(f"Target: 50,000 profiles  |  Output: {output_dir}")
    print(f"{'='*60}\n")

    # ── Step 1: Generate all profile batches ─────────────────────────────
    all_profiles = []

    all_profiles += generate_legitimate_batch(N_LEGITIMATE)
    all_profiles += generate_functional_fraud_batch(N_FUNCTIONAL)
    all_profiles += generate_template_bio_batch(N_TEMPLATE_BIO)
    all_profiles += generate_financial_scam_batch(N_FINANCIAL)
    all_profiles += generate_ring_batch(N_RING_HUBS, N_RING_RELAYS, N_RING_SATELLITES)
    all_profiles += generate_multi_signal_batch(N_MULTI)

    print(f"\nGenerated {len(all_profiles):,} profiles total.")

    # ── Step 2: Shuffle (so fraud isn't in blocks) ───────────────────────
    random.shuffle(all_profiles)

    # ── Step 3: Flatten and build DataFrame ──────────────────────────────
    print("Flattening profiles to tabular format...")
    flat = flatten_for_csv(all_profiles)
    df = pd.DataFrame(flat)
    print(f"DataFrame shape: {df.shape}")

    # ── Step 4: Save full dataset ─────────────────────────────────────────
    csv_path = os.path.join(output_dir, "profiles.csv")
    pq_path  = os.path.join(output_dir, "profiles.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    print(f"Saved: {csv_path}")
    print(f"Saved: {pq_path}")

    # ── Step 5: Temporal split ────────────────────────────────────────────
    print("Performing temporal split (train: months 1-4, val: 5, test: 6)...")
    train, val, test = temporal_split(df.copy())
    train.to_csv(os.path.join(output_dir, "splits", "train.csv"), index=False)
    val.to_csv(os.path.join(output_dir, "splits", "val.csv"),   index=False)
    test.to_csv(os.path.join(output_dir, "splits", "test.csv"),  index=False)
    print(f"  Train: {len(train):,}  |  Val: {len(val):,}  |  Test: {len(test):,}")

    # ── Step 6: Build interaction graph ───────────────────────────────────
    if not skip_graph:
        print("\nBuilding interaction graph (this may take 2-3 minutes)...")
        # Use a sample for graph construction — full 50k * 20 contacts each
        # is ~1M edges which is fine but slow. Adjust sample size if needed.
        graph_df = df[["profile_id", "is_fraud", "fraud_type",
                        "unique_contacts", "ring_role" if "ring_role" in df.columns else "fraud_type"]].copy()

        # ring_role may not be in df if no ring profiles — handle gracefully
        if "ring_role" not in graph_df.columns:
            graph_df["ring_role"] = None

        G, edges_df, stats_df = build_interaction_graph(graph_df)

        edges_path = os.path.join(output_dir, "edges.csv")
        stats_path = os.path.join(output_dir, "graph_stats.csv")
        edges_df.to_csv(edges_path, index=False)
        stats_df.to_csv(stats_path, index=False)
        print(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
        print(f"Saved: {edges_path}")
        print(f"Saved: {stats_path}")
    else:
        print("Skipping graph generation (--skip_graph flag set).")

    
    generate_and_save_reports(all_profiles, output_dir)

    # ── Step 7: Dataset report ────────────────────────────────────────────
    write_dataset_report(df, output_dir)

    print(f"\nDataset generation complete. Files in: {output_dir}/")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate matrimonial fraud synthetic dataset")
    parser.add_argument("--seed",       type=int, default=42,       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./output",help="Output directory")
    parser.add_argument("--skip_graph", action="store_true",         help="Skip graph generation")
    args = parser.parse_args()

    main(seed=args.seed, output_dir=args.output_dir, skip_graph=args.skip_graph)