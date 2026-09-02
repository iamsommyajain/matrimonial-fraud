"""Combined multi-model pipeline for M1 + M2-A and future models."""

from __future__ import annotations

import argparse
import os
import time
from typing import Iterable

import numpy as np
import pandas as pd

import sys

DATASET_ROOT = os.path.dirname(os.path.dirname(__file__))
REPO_ROOT = os.path.dirname(DATASET_ROOT)
sys.path.insert(0, REPO_ROOT)

from dataset_generation.model1.model1 import score_profile
from dataset_generation.model2.run_m2a import _target_labels
from dataset_generation.model2.text_audit import load_and_validate_profiles, save_audit
from dataset_generation.model2.text_anomaly import compute_text_anomaly_scores
from dataset_generation.model2.text_features import fit_tfidf_vectorizer, save_vectorizer, transform_text


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def run_m1(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for profile in df.to_dict(orient="records"):
        result = score_profile(profile, include_profiling=False)
        rows.append(
            {
                "profile_id": result["profile_id"],
                "m1_score": result["functional_risk_score"],
                "m1_risk_level": result["risk_level"],
                "m1_flags": "|".join(result["flags"]),
            }
        )
    return pd.DataFrame(rows)


def run_m2a(df: pd.DataFrame, output_dir: str, max_features: int, n_neighbors: int, seed: int) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    df["m2a_target"] = _target_labels(df)
    fit_start = time.perf_counter()
    vectorizer, _ = fit_tfidf_vectorizer(df, max_features=max_features, seed=seed)
    tfidf_fit_time = time.perf_counter() - fit_start
    save_vectorizer(vectorizer, output_dir)

    transform_start = time.perf_counter()
    tfidf_matrix = transform_text(df, vectorizer)
    tfidf_transform_time = time.perf_counter() - transform_start

    scored = compute_text_anomaly_scores(df, tfidf_matrix, n_neighbors=n_neighbors)
    m2a_df = scored.scores.copy()
    m2a_df["m2a_target"] = df["m2a_target"].values
    runtime = {
        **scored.runtime,
        "tfidf_fit_time_sec": tfidf_fit_time,
        "tfidf_transform_time_sec": tfidf_transform_time,
    }
    return m2a_df, runtime


def fuse_scores(df: pd.DataFrame, model_score_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    scores = [out[col].astype(float).clip(0.0, 1.0) for col in model_score_columns if col in out.columns]
    if not scores:
        out["combined_risk"] = 0.0
        return out
    combined = np.zeros(len(out), dtype=float)
    for score in scores:
        combined = 1.0 - (1.0 - combined) * (1.0 - score.to_numpy())
    out["combined_risk"] = np.clip(combined, 0.0, 1.0)
    return out


def build_combined_report(df: pd.DataFrame, output_dir: str, runtime: dict, score_columns: list[str]) -> str:
    lines = [
        "=" * 78,
        "COMBINED MULTI-MODEL FRAUD REPORT",
        "=" * 78,
        f"Dataset size: {len(df):,}",
        f"Models fused : {', '.join(score_columns)}",
        "",
        "Score Summary",
    ]
    for col in score_columns + ["combined_risk"]:
        if col in df.columns:
            lines.append(
                f"  {col:<16} mean={df[col].mean():.4f} p90={df[col].quantile(0.90):.4f} max={df[col].max():.4f}"
            )
    lines += [
        "",
        "Operational Notes",
        "  combined_risk is computed with a transparent OR-like fusion:",
        "  combined = 1 - Π(1 - model_score)",
        "  This is easy to extend: register another model score column and include it in fusion.",
        "",
        "Runtime",
    ]
    for key, value in runtime.items():
        lines.append(f"  {key:<24} {value:.4f}")
    report = "\n".join(lines)
    with open(os.path.join(output_dir, "combined_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run combined M1 + M2-A pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--models", type=str, default="m1,m2a")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("Starting combined pipeline...", flush=True)
    print(f"Models     : {args.models}", flush=True)
    print(f"Input      : {args.input}", flush=True)
    print(f"Output dir : {args.output_dir}", flush=True)

    df = load_and_validate_profiles(args.input)
    runtime = {}
    if "m2a" in {m.strip().lower() for m in args.models.split(",") if m.strip()}:
        print("Running M2-A...", flush=True)
        save_audit(df, args.output_dir)
        m2a_df, m2a_runtime = run_m2a(df, args.output_dir, args.max_features, args.n_neighbors, args.seed)
        runtime.update({f"m2a_{k}": v for k, v in m2a_runtime.items()})
    else:
        m2a_df = pd.DataFrame({"profile_id": df["profile_id"], "m2a_score": 0.0})

    if "m1" in {m.strip().lower() for m in args.models.split(",") if m.strip()}:
        print("Running M1...", flush=True)
        m1_df = run_m1(df)
    else:
        m1_df = pd.DataFrame({"profile_id": df["profile_id"], "m1_score": 0.0})

    combined = df[["profile_id", "fraud_type", "is_fraud"]].copy()
    combined = combined.merge(m1_df, on="profile_id", how="left")
    combined = combined.merge(m2a_df[["profile_id", "text_risk", "m2a_target"]], on="profile_id", how="left")
    combined = fuse_scores(combined, ["m1_score", "text_risk"])
    combined["combined_rank"] = combined["combined_risk"].rank(method="first", ascending=False)

    combined_path = os.path.join(args.output_dir, "combined_scores.csv")
    combined.to_csv(combined_path, index=False)

    report = build_combined_report(combined, args.output_dir, runtime, ["m1_score", "text_risk"])
    print(report)
    print(f"Saved combined scores to {combined_path}")
    print(f"Saved combined report to {os.path.join(args.output_dir, 'combined_report.txt')}")


if __name__ == "__main__":
    main()
