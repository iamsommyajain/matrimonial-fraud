from __future__ import annotations

from dataset_generation.model1.config import (
    CONFIDENCE_FLOOR,
    CONTRIBUTION_TEMPERATURE,
    MINIMUM_SIGNAL_ACTIVATION,
    SATURATION_ALPHA,
)

from dataset_generation.model1.m1_types import clamp01


def build_rule_contributions(rule_results, configured_weights):
    raw = {}
    activated = {}

    for result in rule_results:

        conf = (
            max(CONFIDENCE_FLOOR, clamp01(result.confidence))
            if result.score > 0
            else clamp01(result.confidence)
        )

        base = (
            clamp01(result.score)
            * conf
            * clamp01(result.evidence_weight)
            * max(0.0, configured_weights.get(result.rule, 0.0))
        )

        if base < MINIMUM_SIGNAL_ACTIVATION:
            activated_value = 0.0
        else:
            tempered = base ** (
                1.0 / max(CONTRIBUTION_TEMPERATURE, 0.01)
            )

            activated_value = clamp01(
                tempered * SATURATION_ALPHA
            )

        raw[result.rule] = base
        activated[result.rule] = activated_value

    return raw, activated