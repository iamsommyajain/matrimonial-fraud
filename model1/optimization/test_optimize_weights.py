import unittest
from unittest.mock import patch

import numpy as np

from ..config import OPTIMIZED_RULE_WEIGHTS
from ..m1_types import RuleResult
from ..scoring import scorer as scorer_module
from ..scoring.scorer import compute_functional_risk_score_detailed
from .optimize_weights import optimize_weights, score_cache


class WeightOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.cache = [
            {
                "profile_id": "legitimate",
                "fraud_type": "legitimate",
                "is_fraud": 0,
                "is_functional_fraud": 0,
                "rule_values": {"rule_a": (0.0, 1.0, 1.0)},
            },
            {
                "profile_id": "fraud",
                "fraud_type": "functional",
                "is_fraud": 1,
                "is_functional_fraud": 1,
                "rule_values": {"rule_a": (1.0, 1.0, 1.0)},
            },
        ]

    def test_zero_weight_removes_rule_signal(self):
        self.assertEqual(score_cache(self.cache, {"rule_a": 0.0}).tolist(), [0.0, 0.0])
        self.assertGreater(score_cache(self.cache, {"rule_a": 1.0})[1], 0.0)

    def test_search_is_reproducible(self):
        weights = {"rule_a": 0.2}
        first = optimize_weights(self.cache, weights, trials=10, seed=7, precision_target=0.5)
        second = optimize_weights(self.cache, weights, trials=10, seed=7, precision_target=0.5)
        self.assertEqual(first, second)
        np.testing.assert_array_equal(score_cache(self.cache, first[0]), score_cache(self.cache, second[0]))

    def test_production_default_uses_optimized_weights(self):
        result = RuleResult(
            rule="age_experience",
            score=0.8,
            confidence=1.0,
            evidence_weight=1.0,
            reason="test",
        )
        with patch.object(scorer_module, "execute_rules", return_value=[result]):
            scored = compute_functional_risk_score_detailed({})

        self.assertEqual(
            scored["score_breakdown"]["configured_weights"]["age_experience"],
            OPTIMIZED_RULE_WEIGHTS["age_experience"],
        )
        self.assertGreater(scored["functional_risk_score"], 0.0)

    def test_explicit_zero_weight_overrides_optimized_default(self):
        result = RuleResult(
            rule="age_experience",
            score=1.0,
            confidence=1.0,
            evidence_weight=1.0,
            reason="test",
        )
        with patch.object(scorer_module, "execute_rules", return_value=[result]):
            scored = compute_functional_risk_score_detailed(
                {}, configured_weights={"age_experience": 0.0}
            )

        self.assertEqual(scored["functional_risk_score"], 0.0)
        self.assertEqual(
            scored["score_breakdown"]["configured_weights"]["age_experience"],
            0.0,
        )

    def test_optimizer_score_matches_production_score(self):
        score = 0.8
        confidence = 0.9
        evidence_weight = 0.75
        cache = [{
            "profile_id": "profile",
            "rule_values": {
                "age_experience": (score, confidence, evidence_weight),
            },
        }]
        expected = score_cache(cache, OPTIMIZED_RULE_WEIGHTS)[0]
        result = RuleResult(
            rule="age_experience",
            score=score,
            confidence=confidence,
            evidence_weight=evidence_weight,
            reason="test",
        )
        with patch.object(scorer_module, "execute_rules", return_value=[result]):
            actual = compute_functional_risk_score_detailed(
                {}, configured_weights=OPTIMIZED_RULE_WEIGHTS
            )["functional_risk_score"]

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()