from __future__ import annotations

import math

from dataset_generation.model1.config import (
    FUSION_STRATEGY,
    SCORE_CAP,
)

from dataset_generation.model1.m1_types import clamp01

from dataset_generation.model1.fusion.attribution import contribution_entropy


def fuse_contributions(activated: dict):

    strategy = FUSION_STRATEGY

    if strategy == "capped_linear":
        total, after, survival_product = _linear_fusion(activated)

    elif strategy == "logit_additive":
        total, after, survival_product = _logit_fusion(activated)

    else:
        total, after, survival_product = _probabilistic_fusion(activated)

    pre_fusion_score = sum(activated.values())

    entropy = contribution_entropy(after)

    active = sum(
        1 for value in activated.values()
        if value > 0
    )

    damping_ratio = (
        total / pre_fusion_score
        if pre_fusion_score > 0
        else 0.0
    )

    diagnostics = {
        "fusion_strategy": strategy,
        "pre_fusion_score": round(pre_fusion_score, 6),
        "post_fusion_score": round(total, 6),
        "damping_ratio": round(damping_ratio, 6),
        "number_of_active_rules": active,
        "contribution_entropy": round(entropy, 6),
        "score_space_utilization": round(total / SCORE_CAP, 6),
        "survival_product": round(survival_product, 6),
    }

    return after, diagnostics


def _linear_fusion(activated):

    total = min(SCORE_CAP, sum(activated.values()))

    denom = sum(activated.values())

    if denom <= 0:
        after = {k: 0.0 for k in activated}
    else:
        after = {
            k: total * v / denom
            for k, v in activated.items()
        }

    survival_product = 1.0 - total

    return total, after, survival_product


def _logit_fusion(activated):

    total_raw = sum(activated.values())

    logit = -3.0 + 5.0 * total_raw

    total = clamp01(
        1.0 / (1.0 + math.exp(-logit))
    )

    total = min(SCORE_CAP, total)

    denom = sum(activated.values())

    if denom <= 0:
        after = {k: 0.0 for k in activated}
    else:
        after = {
            k: total * v / denom
            for k, v in activated.items()
        }

    survival_product = 1.0 - total

    return total, after, survival_product


def _probabilistic_fusion(activated):

    survival_product = 1.0

    cumulative = 0.0

    after = {}

    for rule, value in activated.items():

        before = cumulative

        survival_product *= (1.0 - value)

        cumulative = clamp01(
            1.0 - survival_product
        )

        after[rule] = max(
            0.0,
            cumulative - before
        )

    total = min(SCORE_CAP, cumulative)

    return total, after, survival_product