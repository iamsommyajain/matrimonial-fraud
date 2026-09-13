from __future__ import annotations

import math


def contribution_entropy(contributions: dict):

    total = sum(contributions.values())

    if total <= 0:
        return 0.0

    entropy = 0.0

    active = 0

    for value in contributions.values():

        if value <= 0:
            continue

        active += 1

        p = value / total

        entropy -= p * math.log2(p)

    if active <= 1:
        return 0.0

    return entropy / math.log2(active)