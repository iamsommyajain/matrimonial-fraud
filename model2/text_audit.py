"""Text audit for Model 2A.

This module inspects the distribution of the text fields used by M2-A:
bio, hobbies, and interests.
"""

from __future__ import annotations

from collections import Counter
import os
import re
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"profile_id", "fraud_type", "is_fraud"}
TEXT_COLUMNS = ("bio", "hobbies", "interests")

ALIASES = {
    "bio": ("bio", "bio_text"),
    "hobbies": ("hobbies",),
    "interests": ("interests", "partner_preferences"),
}


def normalize_text(value: object) -> str:
    """Light normalization only: lowercase, trim, whitespace and punctuation spacing."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = str(value).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"([.,!?;:])(?=\S)", r"\1 ", text)
    return text.strip()


def build_combined_text(df: pd.DataFrame) -> pd.Series:
    parts = []
    for col in TEXT_COLUMNS:
        source = col if col in df.columns else next((alias for alias in ALIASES[col] if alias in df.columns), None)
        if source is None:
            parts.append(pd.Series([""] * len(df), index=df.index))
        else:
            parts.append(df[source].fillna("").map(normalize_text))
    return (parts[0] + " " + parts[1] + " " + parts[2]).str.replace(r"\s+", " ", regex=True).str.strip()


def resolve_text_source(df: pd.DataFrame, field: str) -> str | None:
    """Return the first available source column for a canonical text field."""
    if field in df.columns:
        return field
    for alias in ALIASES.get(field, ()):
        if alias in df.columns:
            return alias
    return None


def _duplicate_frequency_summary(counts: Iterable[int]) -> pd.DataFrame:
    counter = Counter(counts)
    rows = [{"duplicate_group_size": k, "profile_count": v} for k, v in sorted(counter.items())]
    return pd.DataFrame(rows)


def _rate(df: pd.DataFrame, mask: pd.Series) -> float:
    if len(df) == 0:
        return 0.0
    return float(mask.mean())


def generate_text_audit(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """Return a human-readable audit text and a CSV-friendly summary table."""
    resolved = {col: resolve_text_source(df, col) for col in TEXT_COLUMNS}
    missing = {
        col: (df[src].isna().sum() if src is not None else len(df))
        for col, src in resolved.items()
    }
    combined = build_combined_text(df)
    combined_len = combined.str.len().fillna(0)
    bio_source = resolved["bio"]
    bio_len = (
        df[bio_source].fillna("").map(normalize_text).str.len().fillna(0)
        if bio_source is not None
        else pd.Series([0] * len(df), index=df.index)
    )

    bio_counts = (
        df[bio_source].fillna("").map(normalize_text).value_counts()
        if bio_source is not None
        else pd.Series(dtype=int)
    )
    if bio_source is not None:
        exact_bio_dup = df[bio_source].fillna("").map(normalize_text).map(bio_counts).fillna(0).astype(int)
    else:
        exact_bio_dup = pd.Series([0] * len(df), index=df.index)
    combined_counts = combined.value_counts()
    exact_combined_dup = combined.map(combined_counts).fillna(0).astype(int)

    legit = df["fraud_type"].fillna("legitimate").eq("legitimate")
    template = df["fraud_type"].fillna("").eq("template_bio")

    lines = [
        "=" * 78,
        "MODEL 2A TEXT AUDIT",
        "=" * 78,
        f"Total profiles: {len(df):,}",
        f"Missing bio: {missing['bio']:,} ({missing['bio'] / len(df):.2%})",
        f"Missing hobbies: {missing['hobbies']:,} ({missing['hobbies'] / len(df):.2%})",
        f"Missing interests: {missing['interests']:,} ({missing['interests'] / len(df):.2%})",
        f"Unique normalized bio count: {bio_counts.size:,}",
        f"Exact duplicate bio rows: {int((exact_bio_dup > 1).sum()):,}",
        "",
        "Most frequent bios:",
    ]
    for text, count in bio_counts.head(10).items():
        lines.append(f"  [{count:>5}] {text[:160]}")
    lines += [
        "",
        f"Bio length mean / median / p90: {bio_len.mean():.2f} / {bio_len.median():.2f} / {bio_len.quantile(0.90):.2f}",
        f"Combined length mean / median / p90: {combined_len.mean():.2f} / {combined_len.median():.2f} / {combined_len.quantile(0.90):.2f}",
        "",
        "Duplicate frequency distribution (combined text):",
    ]
    dup_summary = _duplicate_frequency_summary(combined_counts.value_counts().values)
    if dup_summary.empty:
        lines.append("  none")
    else:
        for _, row in dup_summary.head(20).iterrows():
            lines.append(f"  group_size={int(row['duplicate_group_size'])} profile_count={int(row['profile_count'])}")
    lines += [
        "",
        f"Exact duplicate rate for legitimate profiles: {_rate(df, (exact_combined_dup > 1) & legit):.2%}",
        f"Exact duplicate rate for template_bio profiles: {_rate(df, (exact_combined_dup > 1) & template):.2%}",
        "",
        "Exact duplicate rate by fraud type:",
    ]
    for fraud_type, group in df.groupby(df["fraud_type"].fillna("legitimate")):
        group_combined = combined.loc[group.index]
        rate = float((group_combined.map(combined_counts) > 1).mean()) if len(group) else 0.0
        lines.append(f"  {fraud_type:<24} {rate:.2%} ({len(group):,})")

    audit_text = "\n".join(lines)
    summary = pd.DataFrame(
        {
            "metric": [
                "total_profiles",
                "missing_bio",
                "missing_hobbies",
                "missing_interests",
                "unique_bio_count",
                "exact_duplicate_bio_rows",
                "bio_length_mean",
                "bio_length_median",
                "bio_length_p90",
                "combined_length_mean",
                "combined_length_median",
                "combined_length_p90",
            ],
            "value": [
                len(df),
                int(missing["bio"]),
                int(missing["hobbies"]),
                int(missing["interests"]),
                int(bio_counts.size),
                int((exact_bio_dup > 1).sum()),
                float(bio_len.mean()),
                float(bio_len.median()),
                float(bio_len.quantile(0.90)),
                float(combined_len.mean()),
                float(combined_len.median()),
                float(combined_len.quantile(0.90)),
            ],
        }
    )
    return audit_text, summary


def load_and_validate_profiles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {', '.join(missing)}. "
            "M2-A requires profile_id, fraud_type, and is_fraud, plus at least one text field."
        )
    if not any(col in df.columns for col in ("bio", "bio_text")):
        raise ValueError(f"Missing bio text field in {csv_path}. Expected 'bio' or 'bio_text'.")
    if not any(col in df.columns for col in ("hobbies",)):
        raise ValueError(f"Missing hobbies field in {csv_path}. Expected 'hobbies'.")
    if not any(col in df.columns for col in ("interests", "partner_preferences")):
        print("WARNING: no interests/partner_preferences column found; M2-A will score without an interests signal.")
    return df


def save_audit(df: pd.DataFrame, output_dir: str) -> tuple[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    audit_text, summary = generate_text_audit(df)
    with open(os.path.join(output_dir, "text_audit.txt"), "w", encoding="utf-8") as f:
        f.write(audit_text)
    summary.to_csv(os.path.join(output_dir, "text_audit.csv"), index=False)
    return audit_text, summary
