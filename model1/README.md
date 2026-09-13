# Model 1: Functional Consistency

Model 1 is a deterministic, explainable fraud scorer. It looks for contradictions inside a profile, such as an implausible combination of age, education, profession, income, experience, email, behavior, and text.

## Learning Order

Read the implementation in this order:

1. `config.py` and `m1_types.py`
   - thresholds, rule weights, evidence weights, and result data structures
2. `feature_extraction/feature_extractor.py`
   - converts one raw profile into normalized feature groups
3. `rules/fuzzy_rules.py`
   - defines the individual consistency and anomaly rules
4. `rules/executor.py`
   - executes every registered rule
5. `scoring/contribution_builder.py`
   - applies confidence, evidence, and configured rule weights
6. `scoring/probabilistic_fusion.py`
   - combines rule contributions into one bounded risk score
7. `scoring/uncertainty.py` and `scoring/attribution.py`
   - summarize missingness, confidence, and contribution distribution
8. `scoring/scorer.py`
   - orchestrates rule execution, weighting, fusion, and explanations
9. `api/model1.py`
   - public `score_profile()` API used by the rest of the project
10. `evaluation/evaluate_m1.py`
    - batch scoring, temporal evaluation, threshold selection, and diagnostics

## Runtime Flow

```text
raw profile
  -> feature extraction
  -> registered rules
  -> weighted contributions
  -> probabilistic fusion
  -> risk level, flags, and explanations
  -> evaluation metrics
```

The main aggregation is:

```text
risk = 1 - product(1 - activated_rule_contribution)
```

The result is capped by `SCORE_CAP` and includes per-rule evidence so a score can be explained rather than treated as a black box.

## Public API

```python
from dataset_generation.model1.api.model1 import score_profile

result = score_profile(profile)
```

Important result fields include:

- `functional_risk_score`
- `risk_level`
- `flags`
- `rule_results`
- `top_contributors`
- `uncertainty_summary`
- `score_breakdown`

## Evaluation

From the project parent directory, run:

```powershell
python -m dataset_generation.model1.evaluation.evaluate_m1 --dataset dataset_generation/output/profiles.csv
```

The evaluator selects a threshold on the validation split when available and reports metrics on the temporal test split. Without split files, it reports descriptive full-dataset metrics.

## Scope

M1 primarily targets `functional` fraud and partially targets `multi` fraud. It is not intended to be the primary detector for template-bio, coordinated-ring, or financial-scam fraud; those signals belong to the other models.
