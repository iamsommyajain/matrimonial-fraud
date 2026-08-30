"""Model 2A nearest-neighbor scoring and baseline risk construction."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from dataset_generation.model2.text_audit import build_combined_text, normalize_text


@dataclass
class ScoreArtifacts:
    scores: pd.DataFrame
    runtime: dict[str, float]


def _normalize_01(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    mn, mx = float(x.min()), float(x.max())
    if mx <= mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (x - mn) / (mx - mn)


def _safe_mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def compute_text_anomaly_scores(
    df: pd.DataFrame,
    tfidf_matrix,
    n_neighbors: int = 10,
) -> ScoreArtifacts:
    t0 = time.perf_counter()
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n_neighbors + 1, len(df)))
    nn_fit_start = time.perf_counter()
    nn.fit(tfidf_matrix)
    nn_fit_time = time.perf_counter() - nn_fit_start

    query_start = time.perf_counter()
    distances, indices = nn.kneighbors(tfidf_matrix, return_distance=True)
    query_time = time.perf_counter() - query_start
    similarity = 1.0 - distances

    if "bio" in df.columns:
        bio_source = "bio"
    elif "bio_text" in df.columns:
        bio_source = "bio_text"
    else:
        bio_source = None
    if bio_source is None:
        bio_norm = pd.Series([""] * len(df), index=df.index)
    else:
        bio_norm = df[bio_source].fillna("").map(normalize_text)
    combined_norm = build_combined_text(df)
    bio_counts = bio_norm.value_counts()
    combined_counts = combined_norm.value_counts()

    exact_bio_duplicate_count = bio_norm.map(bio_counts).fillna(0).astype(int) - 1
    exact_combined_text_duplicate_count = combined_norm.map(combined_counts).fillna(0).astype(int) - 1

    rows = []
    for i in range(len(df)):
        neigh_idx = indices[i].tolist()
        neigh_sim = similarity[i].tolist()
        if neigh_idx and neigh_idx[0] == i:
            neigh_idx = neigh_idx[1:]
            neigh_sim = neigh_sim[1:]
        else:
            for j, idx in enumerate(neigh_idx):
                if idx == i:
                    neigh_idx.pop(j)
                    neigh_sim.pop(j)
                    break
        top_sims = np.asarray(neigh_sim[:10], dtype=float)
        rows.append(
            {
                "nearest_profile_1": df.iloc[neigh_idx[0]]["profile_id"] if len(neigh_idx) > 0 else None,
                "nearest_profile_2": df.iloc[neigh_idx[1]]["profile_id"] if len(neigh_idx) > 1 else None,
                "nearest_profile_3": df.iloc[neigh_idx[2]]["profile_id"] if len(neigh_idx) > 2 else None,
                "nearest_similarity_1": float(neigh_sim[0]) if len(neigh_sim) > 0 else 0.0,
                "nearest_similarity_2": float(neigh_sim[1]) if len(neigh_sim) > 1 else 0.0,
                "nearest_similarity_3": float(neigh_sim[2]) if len(neigh_sim) > 2 else 0.0,
                "max_similarity": float(top_sims.max()) if top_sims.size else 0.0,
                "mean_top3_similarity": _safe_mean(top_sims[:3]),
                "mean_top5_similarity": _safe_mean(top_sims[:5]),
                "mean_top10_similarity": _safe_mean(top_sims[:10]),
                "neighbors_above_070": int((np.asarray(neigh_sim) >= 0.70).sum()),
                "neighbors_above_080": int((np.asarray(neigh_sim) >= 0.80).sum()),
                "neighbors_above_090": int((np.asarray(neigh_sim) >= 0.90).sum()),
                "neighbors_above_095": int((np.asarray(neigh_sim) >= 0.95).sum()),
                "exact_combined_text_duplicate_count": int(exact_combined_text_duplicate_count.iloc[i]),
                "exact_bio_duplicate_count": int(exact_bio_duplicate_count.iloc[i]),
            }
        )

    scores = pd.DataFrame(rows)
    norm_neighbors_090 = _normalize_01(scores["neighbors_above_090"])
    norm_neighbors_095 = _normalize_01(scores["neighbors_above_095"])
    norm_duplicates = _normalize_01(scores["exact_combined_text_duplicate_count"] + scores["exact_bio_duplicate_count"])
    raw_text_risk = (
        0.15 * scores["max_similarity"]
        + 0.10 * scores["mean_top3_similarity"]
        + 0.20 * scores["mean_top5_similarity"]
        + 0.20 * norm_neighbors_090
        + 0.20 * norm_neighbors_095
        + 0.15 * norm_duplicates
    ).clip(0.0, 1.0)
    text_risk = (1.0 - raw_text_risk).clip(0.0, 1.0)

    out = pd.concat(
        [
            df[["profile_id", "fraud_type"]].reset_index(drop=True),
            pd.Series(df.get("m2a_target", 0), name="m2a_target").reset_index(drop=True),
            scores.reset_index(drop=True),
            pd.DataFrame({
                "normalized_neighbors_above_090": norm_neighbors_090,
                "normalized_neighbors_above_095": norm_neighbors_095,
                "normalized_duplicate_count": norm_duplicates,
                "raw_text_risk": raw_text_risk,
                "text_risk": text_risk,
            }),
        ],
        axis=1,
    )

    runtime = {
        "nn_fit_time_sec": nn_fit_time,
        "nn_query_time_sec": query_time,
        "tfidf_transform_time_sec": 0.0,
        "total_time_sec": time.perf_counter() - t0,
    }
    return ScoreArtifacts(out, runtime)
