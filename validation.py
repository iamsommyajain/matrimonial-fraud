"""Dataset validation and diagnostic reporting for generated profiles."""

import json
import os

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


GRADUATION_AGE = {
    "10th": 18,
    "12th": 18,
    "Graduate": 22,
    "Post Graduate": 25,
    "PhD": 30,
}


def _ks_stat(legit, fraud, field):
    if field not in legit.columns or field not in fraud.columns:
        return None
    if legit.empty or fraud.empty:
        return None
    return float(ks_2samp(legit[field].astype(float), fraud[field].astype(float)).statistic)


def _impossible_experience_count(df):
    grad_age = df["education_level"].map(GRADUATION_AGE).fillna(18).astype(float)
    max_exp = np.minimum(df["age"].astype(float) - 18, df["age"].astype(float) - grad_age + 2)
    max_exp = np.maximum(max_exp, 0)
    experience = df["years_experience"].astype(float)
    impossible = ((experience > max_exp) | (experience < 0)).sum()
    return int(impossible)


def _salary_tail_analysis(legit, fraud):
    if legit.empty or fraud.empty:
        return {
            "legit_p90": None,
            "legit_p95": None,
            "fraud_above_legit_p90": None,
            "fraud_above_legit_p95": None,
        }
    legit_income = legit["annual_income_lpa"].astype(float)
    fraud_income = fraud["annual_income_lpa"].astype(float)
    legit_p90 = float(np.percentile(legit_income, 90))
    legit_p95 = float(np.percentile(legit_income, 95))
    return {
        "legit_p90": legit_p90,
        "legit_p95": legit_p95,
        "fraud_above_legit_p90": float((fraud_income > legit_p90).mean()),
        "fraud_above_legit_p95": float((fraud_income > legit_p95).mean()),
    }


def _prestige_stack_frequency(df):
    if df.empty:
        return 0.0
    stack = df[
        df["college_tier"].isin(["elite", "premium"]) &
        df["company_tier"].isin(["elite", "upper_mid"]) &
        df["annual_income_lpa"].astype(float).ge(25)
    ]
    return float(len(stack) / len(df))


def validate_dataset(df, output_dir):
    legit = df[df["is_fraud"] == False]
    fraud = df[df["fraud_type"] == "functional"]

    metrics = {
        "ks": {
            "annual_income_lpa": _ks_stat(legit, fraud, "annual_income_lpa"),
            "years_experience": _ks_stat(legit, fraud, "years_experience"),
            "email_consistency_score": _ks_stat(legit, fraud, "email_consistency_score"),
            "messages_sent": _ks_stat(legit, fraud, "messages_sent"),
            "profile_edit_count": _ks_stat(legit, fraud, "profile_edit_count"),
            "unique_contacts": _ks_stat(legit, fraud, "unique_contacts"),
        }
    }

    metrics["impossible_experience_count"] = _impossible_experience_count(df)
    tail = _salary_tail_analysis(legit, fraud)
    metrics.update(tail)
    metrics["prestige_stack_frequency"] = _prestige_stack_frequency(df)
    metrics["fraud_income_std_ratio"] = (
        float(fraud["annual_income_lpa"].astype(float).std()) /
        max(1.0, float(legit["annual_income_lpa"].astype(float).std()))
    ) if not legit.empty and not fraud.empty else None

    warnings = []
    if metrics["impossible_experience_count"] > max(5, len(df) * 0.002):
        warnings.append("High impossible experience attrition (>0.2% of dataset)")
    if metrics["ks"]["annual_income_lpa"] is not None and metrics["ks"]["annual_income_lpa"] > 0.30:
        warnings.append("Functional fraud income separation is strong; consider softening income anomalies.")
    if metrics["ks"]["years_experience"] is not None and metrics["ks"]["years_experience"] > 0.26:
        warnings.append("Functional fraud experience separation is stronger than expected.")
    if metrics["ks"]["messages_sent"] is not None and metrics["ks"]["messages_sent"] < 0.06:
        warnings.append("Behavioral activity is barely distinguishable for functional fraud; consider adding subtle interaction anomalies.")
    if metrics["prestige_stack_frequency"] is not None and metrics["prestige_stack_frequency"] > 0.14:
        warnings.append("High frequency of elite/premium prestige stacking; this may create a synthetic artifact.")
    if tail["fraud_above_legit_p95"] is not None and tail["fraud_above_legit_p95"] > 0.58:
        warnings.append("Too many functional fraud incomes exceed the legit 95th percentile.")

    lines = ["=" * 60, "DATASET VALIDATION REPORT", "=" * 60, ""]
    lines.append(f"Total profiles: {len(df):,}")
    lines.append(f"Legit profiles: {len(legit):,}")
    lines.append(f"Functional fraud profiles: {len(fraud):,}")
    lines.append("")
    lines.append("KS statistics (Functional fraud vs Legit):")
    for name, value in metrics["ks"].items():
        lines.append(f"  {name:<24} {value if value is not None else 'n/a'}")
    lines.append("")
    lines.append("Income tail diagnostics:")
    lines.append(f"  Legit 90th percentile: {tail['legit_p90']:.2f} lpa")
    lines.append(f"  Legit 95th percentile: {tail['legit_p95']:.2f} lpa")
    lines.append(f"  Fraud above legit p90: {tail['fraud_above_legit_p90']:.2%}")
    lines.append(f"  Fraud above legit p95: {tail['fraud_above_legit_p95']:.2%}")
    lines.append(f"  Fraud income std ratio: {metrics['fraud_income_std_ratio']:.2f}")
    lines.append("")
    lines.append(f"Impossible experience count: {metrics['impossible_experience_count']}")
    lines.append(f"Prestige stack frequency: {metrics['prestige_stack_frequency']:.2%}")
    lines.append("")
    lines.append("Warnings:")
    if warnings:
        for warn in warnings:
            lines.append(f"  - {warn}")
    else:
        lines.append("  None")

    report_txt = "\n".join(lines)
    report_path = os.path.join(output_dir, "validation_report.txt")
    with open(report_path, "w") as fp:
        fp.write(report_txt)

    metrics_path = os.path.join(output_dir, "validation_metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(metrics, fp, indent=2)

    print("\n" + report_txt)
    print(f"Saved: {report_path}")
    print(f"Saved: {metrics_path}")
    return metrics
