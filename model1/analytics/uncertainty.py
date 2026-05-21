from __future__ import annotations

from m1_types import clamp01


def build_uncertainty_summary(rule_results):

    low_conf = [
        r.rule
        for r in rule_results
        if r.confidence < 0.5
    ]

    avg_conf = (
        sum(r.confidence for r in rule_results)
        / len(rule_results)
        if rule_results else 0.0
    )

    missingness = 0.0

    uncertainty_only = 0.0

    suspicious_missingness = 0.0

    for r in rule_results:

        if r.rule != "contextual_missingness":
            continue

        missingness += r.missingness_impact

        suspicious_missingness = (
            r.features.get(
                "suspicious_missingness_score",
                0.0,
            )
        )

        uncertainty_only = (
            r.features.get(
                "uncertainty_only_score",
                0.0,
            )
        )

    return {
        "average_rule_confidence":
            round(avg_conf, 4),

        "low_confidence_rules":
            low_conf,

        "missingness_impact":
            round(clamp01(missingness), 4),

        "uncertainty_only_score":
            round(clamp01(uncertainty_only), 4),

        "suspicious_missingness_score":
            round(clamp01(suspicious_missingness), 4),
    }