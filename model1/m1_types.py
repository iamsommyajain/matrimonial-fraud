"""
Typed containers for Model 1.

These dataclasses are intentionally small.  They make rule semantics explicit
without turning the scorer into a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


FeatureDict = dict[str, Any]
RuleFn = Callable[[FeatureDict], "RuleResult"]


@dataclass(frozen=True)
class RuleSpec:
    name: str
    description: str
    configured_weight: float
    evidence_weight: float
    feature_groups: tuple[str, ...]
    fn: RuleFn


@dataclass
class RuleResult:
    rule: str
    score: float
    confidence: float
    evidence_weight: float
    reason: str
    features: dict[str, Any] = field(default_factory=dict)
    missingness_impact: float = 0.0
    fired: bool = False
    abstained_reason: str | None = None
    raw_signal_value: float | int | str | None = None
    normalized_signal_value: float | None = None
    threshold_used: float | int | str | None = None
    effective_weight: float = 0.0
    missing_fields: list[str] = field(default_factory=list)
    missingness_penalty: float = 0.0
    contribution_before_fusion: float = 0.0
    contribution_after_fusion: float = 0.0
    runtime_ms: float = 0.0

    @property
    def weighted_rule_score(self) -> float:
        return clamp01(self.score) * clamp01(self.confidence) * clamp01(self.evidence_weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "score": round(clamp01(self.score), 4),
            "confidence": round(clamp01(self.confidence), 4),
            "evidence_weight": round(clamp01(self.evidence_weight), 4),
            "reason": self.reason,
            "features": self.features,
            "missingness_impact": round(clamp01(self.missingness_impact), 4),
            "fired": bool(self.fired),
            "abstained_reason": self.abstained_reason,
            "raw_signal_value": self.raw_signal_value,
            "normalized_signal_value": (
                round(clamp01(self.normalized_signal_value), 4)
                if self.normalized_signal_value is not None else None
            ),
            "threshold_used": self.threshold_used,
            "effective_weight": round(clamp01(self.effective_weight), 4),
            "missing_fields": self.missing_fields,
            "missingness_penalty": round(clamp01(self.missingness_penalty), 4),
            "contribution_before_fusion": round(clamp01(self.contribution_before_fusion), 4),
            "contribution_after_fusion": round(clamp01(self.contribution_after_fusion), 4),
            "runtime_ms": round(max(0.0, float(self.runtime_ms)), 6),
            "weighted_rule_score": round(self.weighted_rule_score, 4),
        }


@dataclass
class ScoreBreakdown:
    combined_risk: float
    risk_level: str
    survival_product: float
    effective_scores: dict[str, float]
    configured_weights: dict[str, float]
    top_contributors: list[dict[str, Any]]
    uncertainty_summary: dict[str, Any]
    fusion_diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "combined_risk": round(self.combined_risk, 4),
            "risk_level": self.risk_level,
            "survival_product": round(self.survival_product, 6),
            "effective_scores": {
                k: round(v, 4) for k, v in self.effective_scores.items()
            },
            "configured_weights": self.configured_weights,
            "top_contributors": self.top_contributors,
            "uncertainty_summary": self.uncertainty_summary,
            "fusion_diagnostics": self.fusion_diagnostics,
        }


def clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
