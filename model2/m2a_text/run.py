"""Main runner for Model 2A."""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dataset_generation.model2.m2a_text.evaluate import build_concise_report, build_report
from dataset_generation.model2.m2a_text.anomaly import compute_text_anomaly_scores
from dataset_generation.model2.m2a_text.audit import load_and_validate_profiles, save_audit
from dataset_generation.model2.m2a_text.features import fit_tfidf_vectorizer, load_split_or_full, save_vectorizer, transform_text


def _target_labels(df: pd.DataFrame) -> pd.Series:
    fraud_type = df["fraud_type"].fillna("")
    signals = df.get("injected_signals_str", pd.Series([""] * len(df), index=df.index)).fillna("")
    text_template_signal = signals.str.contains("template_bio", case=False, na=False)
    text_template_signal |= signals.str.contains("text_template", case=False, na=False)
    text_template_signal |= signals.str.contains("bio_template", case=False, na=False)
    target = fraud_type.eq("template_bio")
    target |= fraud_type.eq("multi") & text_template_signal
    return target.astype(int)


def main():
    parser = argparse.ArgumentParser(description="Run Model 2A text anomaly detection")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_neighbors", type=int, default=10)
    parser.add_argument("--max_features", type=int, default=20000)
    parser.add_argument("--text_fields", type=str, default="bio", help="Comma-separated fields: bio,hobbies,interests")
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting Model 2A run...", flush=True)
    print(f"Input      : {args.input}", flush=True)
    print(f"Output dir : {args.output_dir}", flush=True)
    print(f"Seed       : {args.seed}", flush=True)
    print(f"Neighbors  : {args.n_neighbors}", flush=True)
    print(f"Max feats  : {args.max_features}", flush=True)
    text_fields = tuple(field.strip() for field in args.text_fields.split(",") if field.strip())
    print(f"Text fields: {', '.join(text_fields)}", flush=True)

    print("Loading dataset and validating schema...", flush=True)
    full_df = load_and_validate_profiles(args.input)
    full_df["m2a_target"] = _target_labels(full_df)
    print("Writing text audit...", flush=True)
    audit_text, _ = save_audit(full_df, args.output_dir)

    print("Detecting split files and preparing TF-IDF source...", flush=True)
    artifacts = load_split_or_full(args.input)
    train_df = artifacts.train_df.copy()
    train_df["m2a_target"] = _target_labels(train_df)

    print("Fitting TF-IDF vectorizer...", flush=True)
    fit_start = time.perf_counter()
    vectorizer, train_tfidf_matrix = fit_tfidf_vectorizer(train_df, max_features=args.max_features, seed=args.seed, text_fields=text_fields)
    tfidf_fit_time = time.perf_counter() - fit_start
    save_vectorizer(vectorizer, args.output_dir)

    print("Transforming text into TF-IDF matrix...", flush=True)
    transform_start = time.perf_counter()
    tfidf_matrix = transform_text(full_df, vectorizer, text_fields=text_fields)
    tfidf_transform_time = time.perf_counter() - transform_start

    print("Computing nearest-neighbor text similarities...", flush=True)
    scores_artifacts = compute_text_anomaly_scores(
        full_df,
        tfidf_matrix,
        n_neighbors=args.n_neighbors,
        reference_df=train_df,
        reference_matrix=train_tfidf_matrix,
        text_fields=text_fields,
    )
    scores_df = scores_artifacts.scores
    scores_df["m2a_target"] = full_df["m2a_target"].values
    scores_df["tfidf_fit_time_sec"] = tfidf_fit_time
    scores_df["tfidf_transform_time_sec"] = tfidf_transform_time
    scores_df["nn_fit_time_sec"] = scores_artifacts.runtime["nn_fit_time_sec"]
    scores_df["nn_query_time_sec"] = scores_artifacts.runtime["nn_query_time_sec"]
    scores_df["total_time_sec"] = scores_artifacts.runtime["total_time_sec"]

    scores_path = os.path.join(args.output_dir, "m2a_scores.csv")
    scores_df.to_csv(scores_path, index=False)

    threshold_source = scores_df
    threshold = 0.5
    if artifacts.val_df is not None and len(artifacts.val_df):
        print("Selecting threshold from validation split...", flush=True)
        if not artifacts.split_dir:
            raise FileNotFoundError("Validation split was detected in memory, but split_dir was not set.")
        val = pd.read_csv(os.path.join(artifacts.split_dir, "val.csv"))
        if "m2a_target" not in val.columns:
            val["m2a_target"] = _target_labels(val)
        val_scores = scores_df[scores_df["profile_id"].isin(val["profile_id"])]
        if not val_scores.empty:
            sweep = []
            y = val_scores["m2a_target"].astype(int).to_numpy()
            s = val_scores["text_risk"].to_numpy()
            for t in np.linspace(0.0, 1.0, 101):
                pred = (s >= t).astype(int)
                tp = int(((y == 1) & (pred == 1)).sum())
                fp = int(((y == 0) & (pred == 1)).sum())
                fn = int(((y == 1) & (pred == 0)).sum())
                precision = tp / (tp + fp) if tp + fp else 0.0
                recall = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                sweep.append((t, precision, recall, f1))
            threshold = max(sweep, key=lambda x: (x[3], x[2], x[1]))[0]
    else:
        print("WARNING: no validation split found; using threshold 0.50 as exploratory fallback.")

    evaluation_df = scores_df
    evaluation_scope = "full_dataset"
    if artifacts.test_df is not None and len(artifacts.test_df):
        test_ids = set(artifacts.test_df["profile_id"])
        evaluation_df = scores_df[scores_df["profile_id"].isin(test_ids)].copy()
        evaluation_scope = "temporal_test"
        scores_df["evaluation_scope"] = scores_df["profile_id"].isin(test_ids).map(
            {True: "temporal_test", False: "train_or_validation_or_reference"}
        )
    else:
        scores_df["evaluation_scope"] = "full_dataset"
    scores_df.to_csv(scores_path, index=False)

    print("Building report...", flush=True)
    report = build_report(scores_df, threshold, args.output_dir, {**scores_artifacts.runtime, "tfidf_fit_time_sec": tfidf_fit_time, "tfidf_transform_time_sec": tfidf_transform_time}, exploratory=artifacts.val_df is None, evaluation_df=evaluation_df, evaluation_scope=evaluation_scope)
    print(build_concise_report(scores_df, threshold, evaluation_df=evaluation_df, evaluation_scope=evaluation_scope))
    print(f"Saved scores to {scores_path}")
    print(f"Saved report to {os.path.join(args.output_dir, 'm2a_report.txt')}")
    print(f"Saved audit to {os.path.join(args.output_dir, 'text_audit.txt')}")

if __name__ == "__main__":
    main()
