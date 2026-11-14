"""CLI for Model 5, GAT-only: train a small 2-layer GAT on the temporal
split, calibrate, and report precision at fixed recall per fraud type next
to the Step 3 degree-only baseline. The Transformer/sequence path (original
Step 5) is parked — it scaled too slowly on this machine's CPU to be
practical right now — so this is GAT vs degree-only, not a 3-way ablation.
Does not wire into the fusion pipeline — that's a separate, later step.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model5.graph_interaction.audit import load_and_validate_edges, load_and_validate_profiles
from model5.graph_interaction.config import PRECISION_AT_RECALL_TARGETS, RANDOM_SEED
from model5.graph_interaction.evaluate import evaluate_group_vs_legit, precision_at_recall
from model5.graph_interaction.features import RAW_FEATURE_COLUMNS, build_node_features, normalize_features
from model5.graph_interaction.graph_builder import build_pyg_data
from model5.graph_interaction.run import _fit_degree_model, _score
from model5.graph_interaction.train import calibrated_scores, train_model

FRAUD_GROUPS = ["functional", "template_bio", "financial_scam", "multi", "coordinated_ring"]

# Maps each edges_audit.csv "topology" tag to one of the 5 network patterns
# from the Model 5 design (plus "no_pattern" for the half of non-ring,
# non-demographic-targeted fraud that uses generic sparse outreach).
# Eval-only: never read for model input, only to group profiles afterward.
PATTERN_TAG_MAP = {
    "ring_star": "star",
    "ring_relay": "relay",
    "ring_satellite_convergence": "ring_satellite",
    "ring_cycle": "ring_satellite",
    "demographic_targeting": "demographic_targeting",
    "isolation_pattern": "isolation",
    "other_fraud_generic": "no_pattern",
}
PATTERN_ORDER = ["star", "ring_satellite", "relay", "demographic_targeting", "isolation", "no_pattern"]
FIVE_PATTERNS = ["star", "ring_satellite", "relay", "demographic_targeting", "isolation"]
# star/relay/ring_satellite only ever appear in a handful of literal rings
# (4 in train, 3 in val, 3 in test) — train is in-sample (the model was
# fit on it), so pooling val+test gives a genuinely held-out evaluation
# with a bit more statistical power than test alone, at the cost of val
# not being fully independent of model *selection* (early stopping used
# it). demographic_targeting/isolation/no_pattern aren't ring-tied and
# have plenty of examples in test alone, so they stay test-only.
POOLED_PATTERNS = {"star", "ring_satellite", "relay"}


def _derive_patterns(edges_audit_path: str, fraud_ids: set) -> pd.Series:
    audit = pd.read_csv(edges_audit_path)
    primary = audit[audit["topology"].isin(PATTERN_TAG_MAP)].copy()
    primary["pattern"] = primary["topology"].map(PATTERN_TAG_MAP)
    dominant = (
        primary.groupby(["source", "pattern"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False)
        .drop_duplicates("source", keep="first")
        .set_index("source")["pattern"]
    )
    pattern_map = dominant.to_dict()
    for pid in fraud_ids - set(dominant.index):
        pattern_map[pid] = "no_pattern"
    return pd.Series(pattern_map, name="pattern")


def _best_f1_threshold(y_val: np.ndarray, s_val: np.ndarray) -> float:
    best_f1, best_t = -1.0, 0.5
    for t in np.linspace(0, 1, 101):
        pred = (s_val >= t).astype(int)
        tp = int(((y_val == 1) & (pred == 1)).sum())
        fp = int(((y_val == 0) & (pred == 1)).sum())
        fn = int(((y_val == 1) & (pred == 0)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def _pattern_rows(profiles_df: pd.DataFrame, patterns: pd.Series, val_ids: set, test_ids: set,
                   scores_by_profile: pd.Series, model_name: str) -> list[dict]:
    df = profiles_df[["profile_id", "is_fraud"]].copy()
    df["score"] = df["profile_id"].map(scores_by_profile)
    df["pattern"] = df["profile_id"].map(patterns)

    val_df = df[df.profile_id.isin(val_ids)]
    threshold = _best_f1_threshold(val_df["is_fraud"].astype(int).to_numpy(), val_df["score"].to_numpy())

    test_only_df = df[df.profile_id.isin(test_ids)]
    pooled_df = df[df.profile_id.isin(val_ids | test_ids)]

    rows = []
    for pattern_name in PATTERN_ORDER:
        pooled = pattern_name in POOLED_PATTERNS
        pool = pooled_df if pooled else test_only_df
        sub = pool[pool.is_fraud & (pool.pattern == pattern_name)]
        if sub.empty:
            continue
        legit_scores = pool[~pool.is_fraud]["score"].to_numpy()
        recall = float((sub["score"] >= threshold).mean())
        pr = precision_at_recall(
            np.concatenate([np.ones(len(sub)), np.zeros(len(legit_scores))]),
            np.concatenate([sub["score"].to_numpy(), legit_scores]), 0.50,
        )
        label = pattern_name + (" (small n)" if pattern_name == "star" else "")
        rows.append({"model": model_name, "pattern": label, "n": len(sub),
                     "pool": "val+test" if pooled else "test",
                     "threshold": round(threshold, 2), "recall_at_threshold": round(recall, 4),
                     "precision_at_recall_50": round(pr["precision"], 4)})
    return rows


def _fixed_fpr_rows(profiles_df: pd.DataFrame, patterns: pd.Series, val_ids: set, test_ids: set,
                     scores_by_profile: pd.Series, model_name: str, fpr: float = 0.05) -> list[dict]:
    """Precision on each of the 5 patterns at a threshold set to exactly
    `fpr` false-positive rate on legit (within the same pool as the
    pattern) — the metric used to judge the two tuning experiments, not AUC."""
    df = profiles_df[["profile_id", "is_fraud"]].copy()
    df["score"] = df["profile_id"].map(scores_by_profile)
    df["pattern"] = df["profile_id"].map(patterns)
    test_only_df = df[df.profile_id.isin(test_ids)]
    pooled_df = df[df.profile_id.isin(val_ids | test_ids)]

    rows = []
    for pattern_name in FIVE_PATTERNS:
        pooled = pattern_name in POOLED_PATTERNS
        pool = pooled_df if pooled else test_only_df
        legit_scores = pool[~pool.is_fraud]["score"].to_numpy()
        threshold = float(np.quantile(legit_scores, 1.0 - fpr))
        sub = pool[pool.is_fraud & (pool.pattern == pattern_name)]
        if sub.empty:
            continue
        tp = int((sub["score"] >= threshold).sum())
        fp = int((legit_scores >= threshold).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        label = pattern_name + (" (small n)" if pattern_name == "star" else "")
        rows.append({"model": model_name, "pattern": label, "n": len(sub),
                     "pool": "val+test" if pooled else "test",
                     f"threshold_at_{int(fpr*100)}pct_fpr": round(threshold, 4),
                     "precision": round(precision, 4), "recall": round(tp / len(sub), 4)})
    return rows


def _group_rows(features_df: pd.DataFrame, test_ids: set, scores_by_profile: pd.Series, model_name: str) -> list[dict]:
    test_df = features_df[features_df["profile_id"].isin(test_ids)].copy()
    test_df["score"] = test_df["profile_id"].map(scores_by_profile)
    legit_scores = test_df[test_df["fraud_type"].isna()]["score"].to_numpy()

    rows = []
    for ft in FRAUD_GROUPS:
        sub = test_df[test_df["fraud_type"] == ft]
        if sub.empty:
            continue
        r = evaluate_group_vs_legit(sub["score"].to_numpy(), legit_scores, PRECISION_AT_RECALL_TARGETS)
        r.update({"model": model_name, "group": ft})
        rows.append(r)
    for name, mask in [
        ("satellites only", (test_df["fraud_type"] == "coordinated_ring") & (test_df["ring_role"] == "satellite")),
        ("hub+relay only", (test_df["fraud_type"] == "coordinated_ring") & (test_df["ring_role"].isin(["hub", "relay"]))),
    ]:
        sub = test_df[mask]
        if sub.empty:
            continue
        r = evaluate_group_vs_legit(sub["score"].to_numpy(), legit_scores, PRECISION_AT_RECALL_TARGETS)
        r.update({"model": model_name, "group": name})
        rows.append(r)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Model 5 GAT-only: train, calibrate, compare to degree baseline")
    parser.add_argument("--profiles", default="./output/profiles.csv")
    parser.add_argument("--edges", default="./output/edges.csv")
    parser.add_argument("--output_dir", default="./model5/outputs")
    parser.add_argument("--splits_dir", default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--max_epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pos_weight", type=float, default=19.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--shuffle_train_labels", action="store_true",
                         help="Sanity check: permute is_fraud within the train split only (val/test untouched). Test AUC should land near 0.5 if there's no leakage.")
    parser.add_argument("--shuffle_val_labels_too", action="store_true",
                         help="Stricter variant: also permute val labels, so early stopping and Platt calibration can't see any true label either. Isolates whether a result survives on raw feature extremeness alone vs. real train-label learning + val-based model selection.")
    parser.add_argument("--drop_features", default="", help="Comma-separated RAW_FEATURE_COLUMNS to zero out before training (ablation).")
    parser.add_argument("--no_skip", action="store_true", help="Disable the raw-feature skip connection (on by default).")
    parser.add_argument("--edges_audit", default=None, help="defaults to <edges_dir>/edges_audit.csv; used only for eval-time pattern grouping, never as model input.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    splits_dir = args.splits_dir or os.path.join(os.path.dirname(args.profiles), "splits")

    print("Loading profiles + edges...", flush=True)
    profiles_df = load_and_validate_profiles(args.profiles)
    edges_df = load_and_validate_edges(args.edges)

    train_ids = set(pd.read_csv(os.path.join(splits_dir, "train.csv"))["profile_id"])
    val_ids = set(pd.read_csv(os.path.join(splits_dir, "val.csv"))["profile_id"])
    test_ids = set(pd.read_csv(os.path.join(splits_dir, "test.csv"))["profile_id"])

    print("Building node features...", flush=True)
    features_df = build_node_features(profiles_df, edges_df)
    scaled_features_df, _ = normalize_features(features_df, train_ids)

    if args.drop_features:
        drop_cols = [c.strip() for c in args.drop_features.split(",") if c.strip()]
        print(f"Ablation: zeroing out features {drop_cols}", flush=True)
        for c in drop_cols:
            scaled_features_df[c] = 0.0

    print("Building PyG Data object...", flush=True)
    build = build_pyg_data(profiles_df, edges_df, scaled_features_df, train_ids, val_ids, test_ids)
    data = build.data
    node_order = build.node_order

    if args.shuffle_train_labels:
        import torch
        g = torch.Generator().manual_seed(args.seed)
        shuffled_y = data.y.clone()
        masks_to_shuffle = [data.train_mask] + ([data.val_mask] if args.shuffle_val_labels_too else [])
        for mask in masks_to_shuffle:
            idx = mask.nonzero(as_tuple=True)[0]
            perm = idx[torch.randperm(len(idx), generator=g)]
            shuffled_y[idx] = data.y[perm]
        data.y = shuffled_y
        scope = "train+val" if args.shuffle_val_labels_too else "train only"
        print(f"Sanity check: shuffled is_fraud within {scope} "
              f"({int(data.train_mask.sum())} train, {int(data.val_mask.sum())} val; "
              f"test untouched: {int(data.test_mask.sum())}).", flush=True)

    in_dim = len(RAW_FEATURE_COLUMNS)
    all_rows = []

    print("\nFitting degree-only baseline...", flush=True)
    train_df_feat = features_df[features_df["profile_id"].isin(train_ids)]
    clf, scaler_deg = _fit_degree_model(train_df_feat, args.seed)
    deg_scores = pd.Series(_score(clf, scaler_deg, features_df), index=features_df["profile_id"])
    all_rows += _group_rows(features_df, test_ids, deg_scores, "degree_baseline")

    use_skip = not args.no_skip
    print(f"\nTraining GAT-only model (2 layers, 4 heads, hidden=64, CPU, skip_connection={use_skip})...", flush=True)
    t0 = time.perf_counter()
    gat_result = train_model(data, None, None, use_sequence=False, in_dim=in_dim, use_skip=use_skip,
                              pos_weight=args.pos_weight, weight_decay=args.weight_decay,
                              max_epochs=args.max_epochs, patience=args.patience, seed=args.seed, device=args.device)
    elapsed = time.perf_counter() - t0
    print(f"  GAT-only: best epoch {gat_result.best_epoch}, {len(gat_result.history)} epochs run, {elapsed:.1f}s", flush=True)
    gat_scores_arr = calibrated_scores(gat_result.model, gat_result.calibrator, data, None, None, device=args.device)
    gat_scores = pd.Series(gat_scores_arr, index=node_order)
    all_rows += _group_rows(features_df, test_ids, gat_scores, "gat_skip" if use_skip else "gat_only")

    print(f"\nTraining no-graph baseline (same MLP classifier + skip features, no GAT/edge_index at all)...", flush=True)
    t0 = time.perf_counter()
    nograph_result = train_model(data, None, None, use_sequence=False, in_dim=in_dim, use_skip=True, use_gat=False,
                                  pos_weight=args.pos_weight, weight_decay=args.weight_decay,
                                  max_epochs=args.max_epochs, patience=args.patience, seed=args.seed, device=args.device)
    elapsed = time.perf_counter() - t0
    print(f"  no-graph: best epoch {nograph_result.best_epoch}, {len(nograph_result.history)} epochs run, {elapsed:.1f}s", flush=True)
    nograph_scores_arr = calibrated_scores(nograph_result.model, nograph_result.calibrator, data, None, None, device=args.device)
    nograph_scores = pd.Series(nograph_scores_arr, index=node_order)
    all_rows += _group_rows(features_df, test_ids, nograph_scores, "no_graph")

    comparison_df = pd.DataFrame(all_rows)
    cols = ["model", "group", "n", "roc_auc", "precision_at_recall_50", "achieved_recall_50",
            "precision_at_recall_80", "achieved_recall_80"]
    comparison_df = comparison_df[cols].sort_values(["group", "model"])
    report_lines = [
        "=" * 100,
        "MODEL 5: degree baseline vs GAT-only",
        "=" * 100,
        "Evaluated on the temporal test split (month 6), calibrated (Platt) GAT scores.",
        "Transformer/sequence path parked — too slow on this machine's CPU; see run.py comments.",
        "",
        comparison_df.to_string(index=False),
    ]
    report = "\n".join(report_lines)
    print("\n" + report, flush=True)

    comparison_df.to_csv(os.path.join(args.output_dir, "m5_gat_vs_degree_metrics.csv"), index=False)
    with open(os.path.join(args.output_dir, "m5_gat_vs_degree_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ── Pattern table: GAT+skip vs no-graph, side by side ──────────────────
    edges_audit_path = args.edges_audit or os.path.join(os.path.dirname(args.edges), "edges_audit.csv")
    fraud_ids = set(profiles_df[profiles_df["is_fraud"]]["profile_id"])
    patterns = _derive_patterns(edges_audit_path, fraud_ids)

    pattern_rows = []
    pattern_rows += _pattern_rows(profiles_df, patterns, val_ids, test_ids, gat_scores, "gat_skip" if use_skip else "gat_only")
    pattern_rows += _pattern_rows(profiles_df, patterns, val_ids, test_ids, nograph_scores, "no_graph")
    pattern_df = pd.DataFrame(pattern_rows).sort_values(["pattern", "model"])

    legit_mask = ~profiles_df.set_index("profile_id").loc[node_order, "is_fraud"].to_numpy()
    legit_stats = []
    for name, arr in [("gat_skip" if use_skip else "gat_only", gat_scores_arr), ("no_graph", nograph_scores_arr)]:
        legit_vals = np.asarray(arr)[legit_mask]
        legit_stats.append({"model": name, "legit_mean": round(float(legit_vals.mean()), 4),
                             "legit_p95": round(float(np.quantile(legit_vals, 0.95)), 4)})

    pattern_report_lines = [
        "=" * 100,
        "MODEL 5: network-pattern table, GAT+skip vs no-graph (same MLP + raw features, no edge_index)",
        "=" * 100,
        "Pattern derived from edges_audit.csv topology tags (eval-only, never a model input).",
        "",
        pattern_df.to_string(index=False),
        "",
        "Legit graph_risk (all splits):",
        pd.DataFrame(legit_stats).to_string(index=False),
    ]
    pattern_report = "\n".join(pattern_report_lines)
    print("\n" + pattern_report, flush=True)
    pattern_df.to_csv(os.path.join(args.output_dir, "m5_pattern_gat_vs_nograph.csv"), index=False)
    with open(os.path.join(args.output_dir, "m5_pattern_gat_vs_nograph_report.txt"), "w", encoding="utf-8") as f:
        f.write(pattern_report)

    # ── Fixed-FPR precision on the 5 patterns — the metric for judging any
    # tuning change, not AUC.
    fpr_rows = []
    fpr_rows += _fixed_fpr_rows(profiles_df, patterns, val_ids, test_ids, gat_scores, "gat_skip" if use_skip else "gat_only")
    fpr_rows += _fixed_fpr_rows(profiles_df, patterns, val_ids, test_ids, nograph_scores, "no_graph")
    fpr_df = pd.DataFrame(fpr_rows).sort_values(["pattern", "model"])
    fpr_report_lines = [
        "=" * 100,
        f"MODEL 5: precision on the 5 patterns at 5% FPR on legit (pos_weight={args.pos_weight}, weight_decay={args.weight_decay})",
        "=" * 100,
        fpr_df.to_string(index=False),
    ]
    fpr_report = "\n".join(fpr_report_lines)
    print("\n" + fpr_report, flush=True)
    fpr_df.to_csv(os.path.join(args.output_dir, "m5_fixed_fpr_precision.csv"), index=False)

    scores_df = pd.DataFrame({"profile_id": node_order, "graph_risk": gat_scores_arr})
    scores_df.to_csv(os.path.join(args.output_dir, "m5_scores.csv"), index=False)
    nograph_scores_df = pd.DataFrame({"profile_id": node_order, "graph_risk": nograph_scores_arr})
    nograph_scores_df.to_csv(os.path.join(args.output_dir, "m5_scores_no_graph.csv"), index=False)
    pd.DataFrame(gat_result.history).to_csv(os.path.join(args.output_dir, "m5_gat_training_history.csv"), index=False)

    print(f"\nSaved report to {args.output_dir}/m5_gat_vs_degree_report.txt", flush=True)
    print(f"Saved metrics to {args.output_dir}/m5_gat_vs_degree_metrics.csv", flush=True)
    print(f"Saved graph_risk scores to {args.output_dir}/m5_scores.csv (NOT yet wired into the fusion pipeline)", flush=True)


if __name__ == "__main__":
    main()
