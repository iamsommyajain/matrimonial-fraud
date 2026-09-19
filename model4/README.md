# Model 4: Community Trust Model

Model 4 is the Truecaller-inspired crowd-signal detector. It does not look at
profile attributes, text, or behavior — it looks at what real users reported
after interacting with a profile, weighs each report by how credible its
reporter is, and aggregates that evidence into a trust risk score that grows
as reports accumulate.

## Learning Order

Read the implementation in this order:

1. `community_trust/config.py`
   - reporter-credibility formula, report-type severity, coordination
     detection thresholds, the single-report corroboration cap, and risk
     bands
2. `community_trust/audit.py`
   - loads and validates `profiles.csv` + `reports.csv`, reports coverage
     (which fraud types actually receive community reports) and orphan/
     schema warnings
3. `community_trust/features.py`
   - turns each raw report row into one credibility-weighted evidence value:
     reporter tenure/verification/flag status → credibility, report_type →
     severity, and an independently recomputed timestamp-clustering check →
     coordination discount
4. `community_trust/aggregation.py`
   - combines a profile's weighted reports with noisy-OR fusion
     (`risk = 1 - Π(1 - weight_i)`), the same fusion style Model 1 uses, so
     independent corroborating reports compound while a lone report cannot
5. `community_trust/evaluate.py`
   - ROC-AUC/PR-AUC, threshold sweep, recall@k, tail-risk concentration,
     fraud-type attribution, coverage stats, and FP/FN case dumps
6. `community_trust/run.py`
   - CLI orchestration; writes everything to `model4/outputs/`

## Design decisions worth knowing

- **Nothing is read from the dataset's precomputed `report_weight` /
  `is_coordinated_report` columns.** Those exist only to make the synthetic
  report metadata look realistic at generation time. Model 4 recomputes its
  own credibility and coordination signals from raw fields
  (`reporter_account_age_days`, `reporter_is_verified`, `reporter_is_flagged`,
  report timestamps) — the weighting logic is owned by the model, the same
  way M1 owns its rule weights and M2 owns its feature engineering.
- **`resolution_status` / `outcome` are never used as inputs.** The generator
  assigns them once per profile based on how many reports it already
  received, so using them as features would leak the label.
- **A single report is capped (`SINGLE_REPORT_WEIGHT_CAP = 0.50`) and can
  never alone push a profile past the "medium" risk band.** Without this cap,
  one report from a verified, long-tenured reporter reaches full credibility
  and can single-handedly drive a profile to ~0.8 risk — which makes the
  model trivially gameable by one false accusation (e.g. a jealous ex).
  Requiring at least two independent, credible reports to reach "high"/
  "critical" is what makes "multiple independent reports compound strongly"
  actually true rather than incidental.
- **Coverage is a known limitation, not a bug.** Model 4 only scores profiles
  that have received at least one report; a profile with zero interaction
  history scores 0 and stays at "low" until someone reports it. It is
  designed to complement M1/M2 (which score every profile at creation time),
  not replace them.

## Runtime Flow

```text
reports.csv + profiles.csv
  -> per-report credibility (age, verification, flag status)
  -> per-report severity (report_type)
  -> per-report coordination discount (recomputed timestamp clustering)
  -> per-report weight, capped for corroboration
  -> per-profile noisy-OR aggregation
  -> community_trust_risk, risk_level, evidence breakdown
  -> evaluation metrics
```

## Running it

```bash
python model4/community_trust/run.py \
  --profiles output/profiles.csv \
  --reports output/reports.csv \
  --output_dir model4/outputs
```

Outputs: `m4_scores.csv` (per-profile score + evidence), `m4_weighted_reports.csv`
(per-report weight breakdown for explainability), `trust_audit.txt`, `m4_report.txt`.

## Evaluation note

An earlier version of `dataset/report_generator.py` guaranteed every fraud
profile 2-10 reports and every legitimate profile at most 1 — which made
"has a report" a near-perfect proxy for `is_fraud` and pushed Model 4 to a
trivial ROC-AUC ≈ 1.0 that reflected the label leaking into the data, not the
model's weighting logic doing anything. That has been fixed:
`report_generator.py` now assigns each fraud profile a report probability by
fraud type (e.g. 82% for `financial_scam`, 45% for `template_bio` — victims
don't always notice or report subtler fraud), and lets a slice of legitimate
profiles receive multiple brigade-style false reports instead of capping at
1. Run `python -m dataset.regenerate_reports` after any change to that logic
to rebuild `output/reports.csv` without re-running the full 50k-profile
generation.

On the regenerated data, Model 4 gets ROC-AUC ≈ 0.82, PR-AUC ≈ 0.65, F1 ≈ 0.74
(P=0.92, R=0.61) on the temporal test split, with 66.9% of fraud profiles
having received any report at all. The 121 test-set false negatives are
almost all zero-report profiles — a coverage limit, not a scoring failure:
M4 cannot flag a profile nobody has reported yet, which is exactly why it is
meant to complement M1/M2 rather than replace them. The 17 false positives
are legitimate profiles hit by 2-3 coordinated false reports, which the
corroboration cap and coordination discount contain but cannot fully
eliminate — a real system would need to weigh this against the cost of a
manual review trigger, not an outright ban.

## Scope

M4 targets all fraud types generically (`is_fraud`), since community reports
are filed regardless of which fraud sub-type is present — unlike M1/M2 which
each target one sub-type. It has zero coverage on fraud types that never
attract reports in this dataset (e.g. `image_theft`, which the current
dataset generator does not actually emit any profiles for).
