"""Data loading, validation, and audit reporting for Model 4."""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

REQUIRED_PROFILE_COLUMNS = {"profile_id", "created_at", "is_fraud", "fraud_type"}
REQUIRED_REPORT_COLUMNS = {
    "report_id",
    "reported_profile_id",
    "reporter_profile_id",
    "report_type",
    "timestamp",
    "reporter_account_age_days",
    "reporter_is_verified",
    "reporter_is_flagged",
}


@dataclass
class TrustAuditResult:
    text: str
    warnings: list[str]


def load_and_validate_profiles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_PROFILE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")
    return df


def load_and_validate_reports(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_REPORT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["reporter_is_verified"] = df["reporter_is_verified"].astype(bool)
    df["reporter_is_flagged"] = df["reporter_is_flagged"].astype(bool)
    df["reporter_account_age_days"] = pd.to_numeric(df["reporter_account_age_days"], errors="coerce").fillna(0.0)
    return df


def generate_trust_audit(profiles_df: pd.DataFrame, reports_df: pd.DataFrame) -> TrustAuditResult:
    all_profile_ids = set(profiles_df["profile_id"])
    reported_ids = set(reports_df["reported_profile_id"])
    reporter_ids = set(reports_df["reporter_profile_id"])

    orphan_reports = reports_df[~reports_df["reported_profile_id"].isin(all_profile_ids)]
    unknown_reporters = reports_df[~reports_df["reporter_profile_id"].isin(all_profile_ids)]

    fraud_lookup = profiles_df.set_index("profile_id")[["is_fraud", "fraud_type"]]
    reports_with_labels = reports_df.join(fraud_lookup, on="reported_profile_id")

    coverage_rows = []
    for fraud_type, group in profiles_df.groupby(profiles_df["fraud_type"].fillna("legitimate")):
        n_profiles = len(group)
        n_with_reports = group["profile_id"].isin(reported_ids).sum()
        coverage_rows.append({
            "fraud_type": fraud_type,
            "profiles": n_profiles,
            "profiles_with_reports": int(n_with_reports),
            "coverage_pct": 100.0 * n_with_reports / n_profiles if n_profiles else 0.0,
            "total_reports": int((reports_df["reported_profile_id"].isin(group["profile_id"])).sum()),
        })
    coverage_df = pd.DataFrame(coverage_rows).sort_values("total_reports", ascending=False)

    report_type_counts = reports_df["report_type"].value_counts()

    lines = [
        "=" * 78,
        "MODEL 4 COMMUNITY TRUST AUDIT",
        "=" * 78,
        f"Total profiles: {len(profiles_df):,}",
        f"Total reports: {len(reports_df):,}",
        f"Distinct reported profiles: {len(reported_ids):,}",
        f"Distinct reporters: {len(reporter_ids):,}",
        f"Orphan reports (reported profile not in profiles.csv): {len(orphan_reports):,}",
        f"Reports from unknown reporters: {len(unknown_reporters):,}",
        "",
        "Report-type distribution:",
    ]
    for report_type, count in report_type_counts.items():
        lines.append(f"  {report_type:<24} {count:>6,}")
    lines += [
        "",
        "Coverage and report volume by fraud type:",
        coverage_df.to_string(index=False),
        "",
        "Reporter credibility inputs:",
        f"  mean reporter_account_age_days: {reports_df['reporter_account_age_days'].mean():.1f}",
        f"  pct verified reporters: {100.0 * reports_df['reporter_is_verified'].mean():.1f}%",
        f"  pct flagged reporters: {100.0 * reports_df['reporter_is_flagged'].mean():.1f}%",
    ]

    warnings = []
    if len(orphan_reports):
        warnings.append(f"{len(orphan_reports)} reports reference a reported_profile_id not present in profiles.csv.")
    zero_coverage_fraud = coverage_df[(coverage_df["fraud_type"] != "legitimate") & (coverage_df["coverage_pct"] == 0.0)]
    if not zero_coverage_fraud.empty:
        warnings.append(
            "Fraud types with zero community-report coverage (M4 cannot flag these on its own): "
            + ", ".join(zero_coverage_fraud["fraud_type"].tolist())
        )

    return TrustAuditResult("\n".join(lines), warnings)


def save_trust_audit(profiles_df: pd.DataFrame, reports_df: pd.DataFrame, output_dir: str) -> TrustAuditResult:
    os.makedirs(output_dir, exist_ok=True)
    result = generate_trust_audit(profiles_df, reports_df)
    with open(os.path.join(output_dir, "trust_audit.txt"), "w", encoding="utf-8") as f:
        f.write(result.text)
    return result
