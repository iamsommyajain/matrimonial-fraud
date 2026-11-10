"""Search Model 1 rule weights without changing the production configuration."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    CONFIDENCE_FLOOR,
    CONTRIBUTION_TEMPERATURE,
    MINIMUM_SIGNAL_ACTIVATION,
    RULE_WEIGHTS,
    SATURATION_ALPHA,
    SCORE_CAP,
)
from ..feature_extraction.feature_extractor import extract_m1_features
from ..rules.executor import execute_rules


FUNCTIONAL_FRAUD_TYPES = frozenset({"functional", "multi"})


def _parse_json_list(value: Any) -> list:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value) and not pd.isna(value)
    return isinstance(value, str) and value.strip().lower() in {"true", "1", "yes", "y"}


def _fraud_type(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "legitimate"
    text = str(value).strip()
    return text.lower() if text and text.lower() not in {"nan", "none", "null"} else "legitimate"


def load_profiles(csv_path: str) -> list[dict]:
    frame = pd.read_csv(csv_path)
    for column in ("login_timestamps", "edit_timestamps", "photo_upload_dates",
                   "login_ip_list", "hobbies", "injected_signals"):
        if column in frame.columns:
            frame[column] = frame[column].map(_parse_json_list)
    return frame.to_dict(orient="records")


def build_rule_cache(profiles: list[dict]) -> list[dict]:
    """Execute extraction and rules once; later trials only apply weights."""
    cache = []
    for profile in profiles:
        results = execute_rules(extract_m1_features(profile))
        cache.append({
            "profile_id": profile.get("profile_id", "unknown"),
            "fraud_type": _fraud_type(profile.get("fraud_type")),
            "is_fraud": int(_truthy(profile.get("is_fraud"))),
            "is_functional_fraud": int(_fraud_type(profile.get("fraud_type")) in FUNCTIONAL_FRAUD_TYPES),
            "rule_values": {
                result.rule: (
                    float(result.score),
                    float(result.confidence),
                    float(result.evidence_weight),
                )
                for result in results
            },
        })
    return cache


def score_cache(cache: list[dict], weights: dict[str, float]) -> np.ndarray:
    scores = []
    for row in cache:
        activated = []
        for rule, (score, confidence, evidence_weight) in row["rule_values"].items():
            confidence = max(CONFIDENCE_FLOOR, min(1.0, confidence)) if score > 0 else min(1.0, confidence)
            base = max(0.0, min(1.0, score)) * confidence * max(0.0, min(1.0, evidence_weight))
            base *= max(0.0, float(weights.get(rule, 0.0)))
            if base >= MINIMUM_SIGNAL_ACTIVATION:
                activated.append(min(
                    1.0,
                    (base ** (1.0 / max(CONTRIBUTION_TEMPERATURE, 0.01))) * SATURATION_ALPHA,
                ))
        survival = 1.0
        for value in activated:
            survival *= 1.0 - value
        scores.append(min(SCORE_CAP, 1.0 - survival))
    # Production scoring exposes the fused score rounded to four decimals.
    return np.round(np.asarray(scores, dtype=float), 4)


def classification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = scores >= threshold
    true_positive = int(((labels == 1) & predicted).sum())
    false_positive = int(((labels == 0) & predicted).sum())
    false_negative = int(((labels == 1) & ~predicted).sum())
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"threshold": float(threshold), "precision": precision, "recall": recall, "f1": f1,
            "tp": true_positive, "fp": false_positive, "fn": false_negative}


def choose_threshold(labels: np.ndarray, scores: np.ndarray, precision_target: float) -> dict[str, float]:
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    metrics = [classification_metrics(labels, scores, float(threshold)) for threshold in candidates]
    eligible = [row for row in metrics if row["precision"] >= precision_target]
    pool = eligible or metrics
    return max(pool, key=lambda row: (row["recall"], row["f1"], row["precision"]))


def _objective(metrics: dict[str, float], precision_target: float) -> float:
    if metrics["precision"] >= precision_target:
        return metrics["recall"]
    return metrics["recall"] - (precision_target - metrics["precision"]) * 2.0


def optimize_weights(
    validation_cache: list[dict],
    current_weights: dict[str, float],
    trials: int,
    seed: int,
    precision_target: float,
) -> tuple[dict[str, float], dict[str, float], list[dict[str, Any]]]:
    rng = random.Random(seed)
    best_weights = dict(current_weights)
    best_metrics = choose_threshold(
        np.asarray([row["is_functional_fraud"] for row in validation_cache]),
        score_cache(validation_cache, best_weights),
        precision_target,
    )
    best_value = _objective(best_metrics, precision_target)
    history = []

    for trial in range(trials):
        candidate = {
            rule: current * math.exp(rng.uniform(math.log(0.25), math.log(4.0)))
            for rule, current in current_weights.items()
        }
        labels = np.asarray([row["is_functional_fraud"] for row in validation_cache])
        metrics = choose_threshold(labels, score_cache(validation_cache, candidate), precision_target)
        value = _objective(metrics, precision_target)
        history.append({"trial": trial, "objective": value, **metrics, **candidate})
        if value > best_value:
            best_value, best_weights, best_metrics = value, candidate, metrics

    return best_weights, best_metrics, history


def _split_cache(cache: list[dict], split_path: Path) -> list[dict]:
    ids = set(pd.read_csv(split_path)["profile_id"])
    return [row for row in cache if row["profile_id"] in ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize Model 1 rule weights on a temporal validation split")
    parser.add_argument("--dataset", default="output/profiles.csv")
    parser.add_argument("--trials", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision-target", type=float, default=0.80)
    parser.add_argument("--output", default="model1/outputs/optimized_weights.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    split_dir = dataset_path.parent / "splits"
    val_path = split_dir / "val.csv"
    test_path = split_dir / "test.csv"
    if not val_path.exists() or not test_path.exists():
        raise SystemExit(f"Temporal validation/test splits are required under {split_dir}")

    cache = build_rule_cache(load_profiles(str(dataset_path)))
    validation_cache = _split_cache(cache, val_path)
    test_cache = _split_cache(cache, test_path)
    if not validation_cache or not test_cache:
        raise SystemExit("Validation and test splits must contain matching profile IDs")

    weights, validation_metrics, history = optimize_weights(
        validation_cache, RULE_WEIGHTS, args.trials, args.seed, args.precision_target
    )
    test_labels = np.asarray([row["is_functional_fraud"] for row in test_cache])
    test_metrics = choose_threshold(test_labels, score_cache(test_cache, weights), args.precision_target)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "objective": f"recall_at_precision_{args.precision_target:.2f}",
        "weights": weights,
        "threshold": validation_metrics["threshold"],
        "validation_metrics": validation_metrics,
        "test_metrics_at_validation_threshold": classification_metrics(
            test_labels, score_cache(test_cache, weights), validation_metrics["threshold"]
        ),
        "test_best_threshold_metrics": test_metrics,
        "optimizer": {"method": "bounded_log_random_search", "trials": args.trials, "seed": args.seed},
    }, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(history).to_csv(output.with_name("weight_optimization_trials.csv"), index=False)
    print(json.dumps({"output": str(output), "validation": validation_metrics,
                      "test_at_validation_threshold": classification_metrics(
                          test_labels, score_cache(test_cache, weights), validation_metrics["threshold"]
                      )}, indent=2))


if __name__ == "__main__":
    main()