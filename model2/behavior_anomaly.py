"""Isolation Forest based behavioral anomaly scoring for Model 2B."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split


@dataclass
class BehaviorScoreArtifacts:
    scores: pd.DataFrame
    model: IsolationForest
    scaler: object
    feature_columns: list[str]
    runtime: dict[str, float]


def _scale_to_unit_interval(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi <= lo:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


def score_behavior(df: pd.DataFrame, X, scaler, feature_columns: list[str], n_estimators: int = 300, random_state: int = 42,
                   novelty_train_mask: pd.Series | None = None, experiment: str = "both") -> BehaviorScoreArtifacts:
    runtime = {}
    if experiment not in {"unsupervised", "novelty", "both"}:
        raise ValueError("experiment must be one of unsupervised, novelty, both")

    if experiment in {"unsupervised", "both"}:
        start = time.perf_counter()
        unsup_model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        unsup_model.fit(X)
        unsup_fit = time.perf_counter() - start
        start = time.perf_counter()
        unsup_raw = -unsup_model.decision_function(X)
        unsup_score = _scale_to_unit_interval(unsup_raw)
        unsup_time = time.perf_counter() - start
    else:
        unsup_model = None
        unsup_fit = unsup_time = 0.0
        unsup_score = np.zeros(len(df), dtype=float)
        unsup_raw = np.zeros(len(df), dtype=float)

    if experiment in {"novelty", "both"}:
        if novelty_train_mask is None:
            novelty_train_mask = df["fraud_type"].fillna("").eq("legitimate")
        novelty_X = X[novelty_train_mask.to_numpy()]
        if len(novelty_X) == 0:
            novelty_X = X
        start = time.perf_counter()
        novelty_model = IsolationForest(
            n_estimators=n_estimators,
            contamination="auto",
            random_state=random_state,
            n_jobs=-1,
        )
        novelty_model.fit(novelty_X)
        nov_fit = time.perf_counter() - start
        start = time.perf_counter()
        nov_raw = -novelty_model.decision_function(X)
        nov_score = _scale_to_unit_interval(nov_raw)
        nov_time = time.perf_counter() - start
    else:
        novelty_model = None
        nov_fit = nov_time = 0.0
        nov_score = np.zeros(len(df), dtype=float)
        nov_raw = np.zeros(len(df), dtype=float)

    if experiment == "unsupervised":
        behavior_risk = unsup_score
    elif experiment == "novelty":
        behavior_risk = nov_score
    else:
        behavior_risk = np.clip(0.6 * unsup_score + 0.4 * nov_score, 0.0, 1.0)

    out = df[["profile_id", "fraud_type", "is_fraud", "m2b_target"]].copy()
    out["raw_isolation_score"] = unsup_raw
    out["raw_novelty_score"] = nov_raw
    out["behavior_risk"] = behavior_risk
    for col in feature_columns:
        out[col] = df[col].values

    runtime["unsupervised_fit_time_sec"] = unsup_fit
    runtime["unsupervised_score_time_sec"] = unsup_time
    runtime["novelty_fit_time_sec"] = nov_fit
    runtime["novelty_score_time_sec"] = nov_time
    runtime["total_time_sec"] = unsup_fit + unsup_time + nov_fit + nov_time

    model = novelty_model if experiment == "novelty" else unsup_model
    return BehaviorScoreArtifacts(out, model, scaler, feature_columns, runtime)

