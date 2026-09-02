"""Model 2A nearest-neighbor scoring and baseline risk construction."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors

from dataset_generation.model2.text_audit import build_combined_text, normalize_text, resolve_text_source


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
    reference_df: pd.DataFrame | None = None,
    reference_matrix=None,
    text_fields=None,
) -> ScoreArtifacts:
    t0 = time.perf_counter()
    reference_df = df if reference_df is None else reference_df
    reference_matrix = tfidf_matrix if reference_matrix is None else reference_matrix
    nn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=min(n_neighbors + 1, len(reference_df)))
    nn_fit_start = time.perf_counter()
    nn.fit(reference_matrix)
    nn_fit_time = time.perf_counter() - nn_fit_start

    query_start = time.perf_counter()
    distances, indices = nn.kneighbors(tfidf_matrix, return_distance=True)
    query_time = time.perf_counter() - query_start
    similarity = 1.0 - distances

    text_fields = tuple(("bio",) if text_fields is None else text_fields)
    bio_source = resolve_text_source(df, "bio") if "bio" in text_fields else None
    if bio_source is None:
        bio_norm = pd.Series([""] * len(df), index=df.index)
    else:
        bio_norm = df[bio_source].fillna("").map(normalize_text)
    combined_norm = build_combined_text(df, text_fields)
    reference_bio = (reference_df[bio_source].fillna("").map(normalize_text)
                     if bio_source is not None and bio_source in reference_df.columns
                     else pd.Series([""] * len(reference_df)))
    reference_combined = build_combined_text(reference_df, text_fields)
    bio_counts = reference_bio.value_counts()
    combined_counts = reference_combined.value_counts()
    exact_bio_duplicate_count = bio_norm.map(bio_counts).fillna(0).astype(int)
    exact_combined_text_duplicate_count = combined_norm.map(combined_counts).fillna(0).astype(int)
    reference_ids_set = set(reference_df["profile_id"])
    exact_bio_duplicate_count -= df["profile_id"].isin(reference_ids_set).astype(int)
    exact_combined_text_duplicate_count -= df["profile_id"].isin(reference_ids_set).astype(int)
    exact_bio_duplicate_count = exact_bio_duplicate_count.clip(lower=0)
    exact_combined_text_duplicate_count = exact_combined_text_duplicate_count.clip(lower=0)

    rows = []
    reference_ids = reference_df["profile_id"].tolist()
    for i in range(len(df)):
        neigh_idx = indices[i].tolist()
        neigh_sim = similarity[i].tolist()
        query_id = df.iloc[i]["profile_id"]
        for j in reversed([j for j, idx in enumerate(neigh_idx) if reference_ids[idx] == query_id]):
            neigh_idx.pop(j)
            neigh_sim.pop(j)
        top_sims = np.asarray(neigh_sim[:10], dtype=float)
        rows.append(
            {
                "nearest_profile_1": reference_df.iloc[neigh_idx[0]]["profile_id"] if len(neigh_idx) > 0 else None,
                "nearest_profile_2": reference_df.iloc[neigh_idx[1]]["profile_id"] if len(neigh_idx) > 1 else None,
                "nearest_profile_3": reference_df.iloc[neigh_idx[2]]["profile_id"] if len(neigh_idx) > 2 else None,
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
    neighbor_denominator = max(1, min(n_neighbors, len(reference_df)))
    norm_neighbors_090 = (scores["neighbors_above_090"] / neighbor_denominator).clip(0.0, 1.0)
    norm_neighbors_095 = (scores["neighbors_above_095"] / neighbor_denominator).clip(0.0, 1.0)
    duplicate_count = scores["exact_combined_text_duplicate_count"] + scores["exact_bio_duplicate_count"]
    reference_max_duplicate = max(1, int((combined_counts.max() if not combined_counts.empty else 1) - 1))
    norm_duplicates = (np.log1p(duplicate_count) / np.log1p(reference_max_duplicate)).clip(0.0, 1.0)
    # Similarity is evidence of repetition; novelty is the complementary
    # signal used by the temporal train-only baseline. Keeping both explicit
    # prevents callers from confusing similarity with fraud risk.
    text_similarity_evidence = (
        0.15 * scores["max_similarity"]
        + 0.10 * scores["mean_top3_similarity"]
        + 0.20 * scores["mean_top5_similarity"]
        + 0.20 * norm_neighbors_090
        + 0.20 * norm_neighbors_095
        + 0.15 * norm_duplicates
    ).clip(0.0, 1.0)
    text_novelty_risk = (1.0 - text_similarity_evidence).clip(0.0, 1.0)
    text_risk = text_novelty_risk

    out = pd.concat(
        [
            df[["profile_id", "fraud_type"]].reset_index(drop=True),
            pd.Series(df.get("m2a_target", 0), name="m2a_target").reset_index(drop=True),
            scores.reset_index(drop=True),
            pd.DataFrame({
                "normalized_neighbors_above_090": norm_neighbors_090,
                "normalized_neighbors_above_095": norm_neighbors_095,
                "normalized_duplicate_count": norm_duplicates,
                "text_similarity_evidence": text_similarity_evidence,
                "text_novelty_risk": text_novelty_risk,
                # Backward-compatible alias used by existing evaluators.
                "raw_text_risk": text_novelty_risk,
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
