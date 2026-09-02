"""Behavioral audit for Model 2-B."""

from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"profile_id", "created_at", "is_fraud", "fraud_type"}
BEHAVIOR_LIST_COLUMNS = ("login_timestamps", "edit_timestamps", "photo_upload_dates", "login_ip_list")
BEHAVIOR_COUNT_COLUMNS = ("messages_sent", "messages_received", "match_requests_sent", "match_accepts", "unique_contacts")

BEHAVIOR_SIGNAL_TARGETS = {
    "burst_login_pattern",
    "rapid_photo_upload",
    "excessive_edits",
    "impossible_geo_logins",
    "behavioral_imbalance",
    "high_outreach_low_response",
    "sudden_inactivity_post_reports",
}


@dataclass
class BehaviorAuditResult:
    text: str
    summary: pd.DataFrame
    warnings: list[str]
    available_signals: dict[str, bool]


def _parse_iso_list(value: object) -> list[datetime]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            import json
            raw = json.loads(text)
        except Exception:
            return []
    else:
        return []
    out = []
    for item in raw:
        try:
            out.append(datetime.fromisoformat(str(item)))
        except Exception:
            continue
    return sorted(out)


def _resolve_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _count_rate(df: pd.DataFrame, mask: pd.Series) -> float:
    return float(mask.mean()) if len(df) else 0.0


def load_and_validate_profiles(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(missing)}")
    return df


def generate_behavior_audit(df: pd.DataFrame) -> BehaviorAuditResult:
    resolved = {col: _resolve_col(df, col) for col in BEHAVIOR_LIST_COLUMNS}
    created_at = pd.to_datetime(df["created_at"], errors="coerce")
    last_active = pd.to_datetime(df["last_active_at"], errors="coerce") if "last_active_at" in df.columns else pd.NaT

    parsed = {col: df[col].map(_parse_iso_list) if col in df.columns else pd.Series([[]] * len(df), index=df.index)
              for col in BEHAVIOR_LIST_COLUMNS}
    event_counts = pd.Series(0, index=df.index, dtype=float)
    for col in BEHAVIOR_LIST_COLUMNS:
        event_counts = event_counts.add(parsed[col].map(len), fill_value=0)

    has_any_behavior = event_counts > 0
    has_device = df["device_type"].notna() if "device_type" in df.columns else pd.Series(False, index=df.index)
    has_location = parsed["login_ip_list"].map(len) > 0
    missing_timestamps = sum(parsed[col].map(len).eq(0).sum() for col in BEHAVIOR_LIST_COLUMNS)

    rows = []
    for fraud_type, group in df.groupby(df["fraud_type"].fillna("legitimate")):
        group_idx = group.index
        rows.append({
            "fraud_type": fraud_type,
            "count": len(group),
            "profiles_with_behavior": int(has_any_behavior.loc[group_idx].sum()),
            "mean_events": float(event_counts.loc[group_idx].mean()) if len(group) else 0.0,
            "mean_logins": float(parsed["login_timestamps"].loc[group_idx].map(len).mean()) if len(group) else 0.0,
            "mean_edits": float(parsed["edit_timestamps"].loc[group_idx].map(len).mean()) if len(group) else 0.0,
            "mean_uploads": float(parsed["photo_upload_dates"].loc[group_idx].map(len).mean()) if len(group) else 0.0,
        })

    lines = [
        "=" * 78,
        "MODEL 2B BEHAVIOR AUDIT",
        "=" * 78,
        f"Total profiles: {len(df):,}",
        f"Profiles with behavioral records: {int(has_any_behavior.sum()):,}",
        f"Total behavioral events: {int(event_counts.sum()):,}",
        f"Profiles with device data: {int(has_device.sum()) if isinstance(has_device, pd.Series) else 0:,}",
        f"Profiles with location data: {int(has_location.sum()):,}",
        f"Profiles with missing timestamps across event lists: {int(missing_timestamps):,}",
        "",
        "Event-type distribution:",
    ]
    for col in BEHAVIOR_LIST_COLUMNS:
        counts = parsed[col].map(len)
        lines.append(f"  {col:<22} total={int(counts.sum()):,} profiles={int((counts > 0).sum()):,}")
    lines += [
        "",
        "Behavior by fraud type:",
    ]
    for _, row in pd.DataFrame(rows).sort_values("count", ascending=False).iterrows():
        lines.append(
            f"  {row['fraud_type']:<24} count={int(row['count']):,} "
            f"with_behavior={int(row['profiles_with_behavior']):,} mean_events={row['mean_events']:.2f}"
        )

    available = {
        "timestamps": any(col in df.columns for col in BEHAVIOR_LIST_COLUMNS),
        "device": "device_type" in df.columns,
        "location": "login_ip_list" in df.columns,
        "behavior_signals": "injected_signals_str" in df.columns,
    }
    warnings = []
    if not available["timestamps"]:
        warnings.append("No behavioral timestamp lists found; M2-B cannot compute temporal features.")
    if not available["location"]:
        warnings.append("No login_ip_list found; location-switch features will be unavailable.")
    if not available["device"]:
        warnings.append("No device_type field found; device-switch features will be unavailable.")

    audit_text = "\n".join(lines)
    summary = pd.DataFrame(rows)
    return BehaviorAuditResult(audit_text, summary, warnings, available)


def save_behavior_audit(df: pd.DataFrame, output_dir: str) -> BehaviorAuditResult:
    os.makedirs(output_dir, exist_ok=True)
    result = generate_behavior_audit(df)
    with open(os.path.join(output_dir, "behavior_audit.txt"), "w", encoding="utf-8") as f:
        f.write(result.text)
    result.summary.to_csv(os.path.join(output_dir, "behavior_audit.csv"), index=False)
    return result

