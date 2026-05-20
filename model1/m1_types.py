"""
Typed containers for Model 1.

These dataclasses are intentionally small.  They make rule semantics explicit
without turning the scorer into a framework.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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

    @property
    def weighted_rule_score(self) -> float:
        return clamp01(self.score) * clamp01(self.confidence) * clamp01(self.evidence_weight)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["score"] = round(clamp01(self.score), 4)
        data["confidence"] = round(clamp01(self.confidence), 4)
        data["evidence_weight"] = round(clamp01(self.evidence_weight), 4)
        data["missingness_impact"] = round(clamp01(self.missingness_impact), 4)
        data["weighted_rule_score"] = round(self.weighted_rule_score, 4)
        return data


@dataclass
class ScoreBreakdown:
    combined_risk: float
    risk_level: str
    survival_product: float
    effective_scores: dict[str, float]
    configured_weights: dict[str, float]
    top_contributors: list[dict[str, Any]]
    uncertainty_summary: dict[str, Any]

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
        }


def clamp01(value: float | int | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
