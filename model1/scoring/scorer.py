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


from ..config import (
    PAIR_FLAG_THRESHOLD,
    RISK_THRESHOLDS,
    RULE_WEIGHTS,
)

from ..m1_types import RuleResult, ScoreBreakdown, clamp01

from ..rules.executor import (
    execute_rules,
    execute_rules_profiled,
)

from ..rules.fuzzy_rules import RULE_REGISTRY

from .contribution_builder import (
    build_rule_contributions,
)

from .probabilistic_fusion import (
    fuse_contributions,
)


from .uncertainty import (
    build_uncertainty_summary,
)


def compute_pair_scores(features: dict) -> dict:
    """
    Backward-compatible name.

    Returns raw rule severity scores by rule name, not final contributions.
    """
    return {
    result.rule: result.score
    for result in execute_rules(features)
}


def compute_functional_risk_score_detailed(features: dict, collect_profiling: bool = False) -> dict:
    if collect_profiling:
        rule_results, rule_runtimes_ms = execute_rules_profiled(features)
    else:
        rule_results = execute_rules(features)
        rule_runtimes_ms = {}
    configured_weights = {
        spec.name: spec.configured_weight
        for spec in RULE_REGISTRY
    }

    raw_contributions, activated_contributions = build_rule_contributions(
        rule_results,
        RULE_WEIGHTS
    )

    effective_scores, fusion_diagnostics = fuse_contributions(activated_contributions)

    combined_risk = round(fusion_diagnostics["post_fusion_score"], 4)
    survival_product = fusion_diagnostics.get("survival_product", 1.0 - combined_risk)

    for result in rule_results:
        result.effective_weight = configured_weights.get(result.rule, 0.0)
        result.contribution_before_fusion = raw_contributions.get(
    result.rule,
    0.0
)
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

def _classify_risk(score: float) -> str:
    for level, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score < hi:
            return level
    return "critical"
