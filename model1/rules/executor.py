from __future__ import annotations

import time

from fuzzy_rules import RULE_REGISTRY


def execute_rules(features: dict):
    return [spec.fn(features) for spec in RULE_REGISTRY]


def execute_rules_profiled(features: dict):
    results = []
    runtimes = {}

    for spec in RULE_REGISTRY:
        start = time.perf_counter()

        result = spec.fn(features)

        runtime_ms = (time.perf_counter() - start) * 1000

        result.runtime_ms = runtime_ms

        runtimes[spec.name] = runtime_ms
        results.append(result)

    return results, runtimes