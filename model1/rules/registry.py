"""
M1 deterministic evidence fusion.

Aggregation:
    combined_risk = 1 - product(1 - weighted_rule_score_i)

where:
    weighted_rule_score_i =
        score_i * confidence_i * evidence_weight_i * configured_weight_i

This is bounded, monotonic, explainable, and gives natural diminishing returns
as independent evidence accumulates.
"""

from __future__ import annotations

import math
import time

from ..config import (
    CONFIDENCE_FLOOR,
    CONTRIBUTION_TEMPERATURE,
    FUSION_STRATEGY,
    MINIMUM_SIGNAL_ACTIVATION,
    PAIR_FLAG_THRESHOLD,
    RISK_THRESHOLDS,
    SATURATION_ALPHA,
    SCORE_CAP,
)
from .fuzzy_rules import RULE_REGISTRY
from ..m1_types import RuleResult, ScoreBreakdown, clamp01


def compute_rule_results(features: dict) -> list[RuleResult]:
    return [spec.fn(features) for spec in RULE_REGISTRY]


def compute_rule_results_profiled(features: dict) -> tuple[list[RuleResult], dict[str, float]]:
    results = []
    runtimes = {}
    for spec in RULE_REGISTRY:
        started = time.perf_counter()
        result = spec.fn(features)
        runtime_ms = (time.perf_counter() - started) * 1000
        result.runtime_ms = runtime_ms
        runtimes[spec.name] = runtime_ms
        results.append(result)
    return results, runtimes


def compute_pair_scores(features: dict) -> dict:
    """
    Backward-compatible name.

    Returns raw rule severity scores by rule name, not final contributions.
    """
    return {result.rule: result.score for result in compute_rule_results(features)}


def compute_functional_risk_score_detailed(features: dict, collect_profiling: bool = False) -> dict:
    if collect_profiling:
        rule_results, rule_runtimes_ms = compute_rule_results_profiled(features)
    else:
        rule_results = compute_rule_results(features)
        rule_runtimes_ms = {}
    configured_weights = {spec.name: spec.configured_weight for spec in RULE_REGISTRY}

    effective_scores, fusion_diagnostics = _compute_fused_contributions(
        rule_results, configured_weights
    )
    combined_risk = round(fusion_diagnostics["post_fusion_score"], 4)
    survival_product = fusion_diagnostics.get("survival_product", 1.0 - combined_risk)

    for result in rule_results:
        result.effective_weight = configured_weights.get(result.rule, 0.0)
        result.contribution_before_fusion = fusion_diagnostics["raw_contributions"].get(result.rule, 0.0)
        result.contribution_after_fusion = effective_scores.get(result.rule, 0.0)

    risk_level = _classify_risk(combined_risk)
    flags = [
        result.rule for result in rule_results
        if result.score >= PAIR_FLAG_THRESHOLD or effective_scores[result.rule] >= 0.06
    ]

    top_contributors = _top_contributors(rule_results, effective_scores, limit=5)
    uncertainty_summary = build_uncertainty_summary(rule_results)
    breakdown = ScoreBreakdown(
        combined_risk=combined_risk,
        risk_level=risk_level,
        survival_product=survival_product,
        effective_scores=effective_scores,
        configured_weights=configured_weights,
        top_contributors=top_contributors,
        uncertainty_summary=uncertainty_summary,
        fusion_diagnostics=fusion_diagnostics,
    )

    return {
        "functional_risk_score": combined_risk,
        "risk_level": risk_level,
        "rule_results": [r.to_dict() for r in rule_results],
        "pair_scores": {r.rule: round(r.score, 4) for r in rule_results},
        "flags": flags,
        "score_breakdown": breakdown.to_dict(),
        "top_contributors": top_contributors,
        "uncertainty_summary": uncertainty_summary,
        "profiling": {"rule_runtimes_ms": rule_runtimes_ms} if collect_profiling else {},
    }


def compute_functional_risk_score(features: dict) -> tuple:
    """
    Backward-compatible tuple API:
        (functional_risk_score, pair_scores, flags, risk_level)
    """
    detailed = compute_functional_risk_score_detailed(features)
    return (
        detailed["functional_risk_score"],
        detailed["pair_scores"],
        detailed["flags"],
        detailed["risk_level"],
    )


def _top_contributors(rule_results: list[RuleResult], effective_scores: dict[str, float],
                      limit: int = 5) -> list[dict]:
    by_rule = {r.rule: r for r in rule_results}
    top = []
    for rule, effective in sorted(effective_scores.items(), key=lambda item: item[1], reverse=True):
        if effective <= 0:
            continue
        result = by_rule[rule]
        top.append({
            "rule": rule,
            "effective_score": round(effective, 4),
            "contribution_before_fusion": round(result.contribution_before_fusion, 4),
            "contribution_after_fusion": round(result.contribution_after_fusion, 4),
            "score": round(result.score, 4),
            "confidence": round(result.confidence, 4),
            "evidence_weight": round(result.evidence_weight, 4),
            "reason": result.reason,
            "features": result.features,
        })
        if len(top) >= limit:
            break
    return top


def _uncertainty_summary(rule_results: list[RuleResult]) -> dict:
    low_conf = [r for r in rule_results if r.confidence < 0.5]
    missing_impact = sum(r.missingness_impact for r in rule_results if r.rule == "contextual_missingness")
    suspicious_missingness = 0.0
    uncertainty_only = 0.0
    for r in rule_results:
        if r.rule == "contextual_missingness":
            suspicious_missingness = r.features.get("suspicious_missingness_score", 0.0)
            uncertainty_only = r.features.get("uncertainty_only_score", 0.0)
    avg_conf = (
        sum(r.confidence for r in rule_results) / len(rule_results)
        if rule_results else 0.0
    )
    return {
        "average_rule_confidence": round(avg_conf, 4),
        "low_confidence_rules": [r.rule for r in low_conf],
        "missingness_impact": round(clamp01(missing_impact), 4),
        "uncertainty_only_score": round(clamp01(uncertainty_only), 4),
        "suspicious_missingness_score": round(clamp01(suspicious_missingness), 4),
    }


def _compute_fused_contributions(rule_results: list[RuleResult],
                                 configured_weights: dict[str, float]) -> tuple[dict[str, float], dict]:
    raw: dict[str, float] = {}
    activated: dict[str, float] = {}
    for result in rule_results:
        conf = max(CONFIDENCE_FLOOR, clamp01(result.confidence)) if result.score > 0 else clamp01(result.confidence)
        base = (
            clamp01(result.score)
            * conf
            * clamp01(result.evidence_weight)
            * max(0.0, configured_weights.get(result.rule, 0.0))
        )
        if base < MINIMUM_SIGNAL_ACTIVATION:
            activated_value = 0.0
        else:
            tempered = base ** (1.0 / max(CONTRIBUTION_TEMPERATURE, 0.01))
            activated_value = clamp01(tempered * SATURATION_ALPHA)
        raw[result.rule] = base
        activated[result.rule] = activated_value

    strategy = FUSION_STRATEGY
    if strategy == "capped_linear":
        total = min(SCORE_CAP, sum(activated.values()))
        after = _allocate_linear(activated, total)
        survival_product = 1.0 - total
    elif strategy == "logit_additive":
        total_raw = sum(activated.values())
        logit = -3.0 + 5.0 * total_raw
        total = clamp01(1.0 / (1.0 + math.exp(-logit)))
        total = min(SCORE_CAP, total)
        after = _allocate_linear(activated, total)
        survival_product = 1.0 - total
    else:
        survival_product = 1.0
        cumulative = 0.0
        after = {}
        for rule, value in activated.items():
            before = cumulative
            survival_product *= (1.0 - value)
            cumulative = clamp01(1.0 - survival_product)
            after[rule] = max(0.0, cumulative - before)
        total = min(SCORE_CAP, cumulative)
        if cumulative > SCORE_CAP and cumulative > 0:
            scale = SCORE_CAP / cumulative
            after = {rule: value * scale for rule, value in after.items()}
            survival_product = 1.0 - SCORE_CAP

    pre_fusion_score = sum(raw.values())
    post_fusion_score = clamp01(total)
    entropy = _contribution_entropy(after)
    active = sum(1 for value in activated.values() if value > 0)
    damping_ratio = post_fusion_score / pre_fusion_score if pre_fusion_score > 0 else 0.0
    diagnostics = {
        "fusion_strategy": strategy,
        "pre_fusion_score": round(pre_fusion_score, 6),
        "post_fusion_score": round(post_fusion_score, 6),
        "damping_ratio": round(damping_ratio, 6),
        "number_of_active_rules": active,
        "contribution_entropy": round(entropy, 6),
        "score_space_utilization": round(post_fusion_score / SCORE_CAP, 6),
        "saturation_alpha": SATURATION_ALPHA,
        "contribution_temperature": CONTRIBUTION_TEMPERATURE,
        "confidence_floor": CONFIDENCE_FLOOR,
        "minimum_signal_activation": MINIMUM_SIGNAL_ACTIVATION,
        "survival_product": round(survival_product, 6),
        
        "activated_contributions": activated,
    }
    return after, diagnostics


def _allocate_linear(activated: dict[str, float], total: float) -> dict[str, float]:
    denom = sum(activated.values())
    if denom <= 0:
        return {rule: 0.0 for rule in activated}
    return {rule: total * value / denom for rule, value in activated.items()}


def _contribution_entropy(contributions: dict[str, float]) -> float:
    total = sum(contributions.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in contributions.values():
        if value <= 0:
            continue
        p = value / total
        entropy -= p * math.log2(p)
    max_entropy = math.log2(sum(1 for value in contributions.values() if value > 0) or 1)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def _classify_risk(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"
