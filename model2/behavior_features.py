"""Behavioral feature engineering for Model 2-B."""

from __future__ import annotations

import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

BEHAVIOR_SIGNAL_TARGETS = {
    "burst_login_pattern",
    "rapid_photo_upload",
    "excessive_edits",
    "impossible_geo_logins",
    "behavioral_imbalance",
    "high_outreach_low_response",
    "sudden_inactivity_post_reports",
}

# The current schema has one device_type value per profile, not device history.
# Keep the derived columns in the exported features for auditability, but do
# not pass constant device features to the anomaly detector.
EXCLUDED_MODEL_FEATURES = frozenset({
    "unique_devices",
    "device_switch_count",
    "device_switch_rate",
    "dominant_device_ratio",
})


def _parse_list(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str) and value.strip():
        try:
            raw = json.loads(value)
        except Exception:
            return []
    else:
        return []
    out = []
    for item in raw:
        try:
            out.append(datetime.fromisoformat(str(item)))
        except Exception:
            pass
    return sorted(out)


def _to_naive_timestamp(value):
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return pd.NaT
    if getattr(ts, "tzinfo", None) is not None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts


def _safe_div(a, b):
    return float(a / b) if b and b != 0 else 0.0


def _burstiness(values):
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    std = arr.std(ddof=0)
    denom = std + mean
    if denom == 0:
        return 0.0
    return float((std - mean) / denom)


def _event_spacings(times):
    if len(times) < 2:
        return np.array([], dtype=float)
    deltas = np.diff(np.array([t.timestamp() for t in times], dtype=float))
    return deltas[deltas >= 0]


def build_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        created_at = _to_naive_timestamp(row.get("created_at"))
        last_active = _to_naive_timestamp(row.get("last_active_at"))
        login_times = _parse_list(row.get("login_timestamps"))
        edit_times = _parse_list(row.get("edit_timestamps"))
        upload_times = _parse_list(row.get("photo_upload_dates"))
        all_times = sorted(login_times + edit_times + upload_times)
        spacings = _event_spacings(all_times)

        if pd.isna(created_at):
            account_age_days = 0.0
        else:
            ref = last_active if not pd.isna(last_active) else pd.Timestamp.utcnow().tz_localize(None)
            account_age_days = max(0.0, float((ref - created_at).total_seconds() / 86400.0))

        def per_day(count):
            return _safe_div(count, max(account_age_days, 1.0))

        weekend_ratio = float(pd.Series([t.weekday() >= 5 for t in all_times]).mean()) if all_times else 0.0
        night_ratio = float(pd.Series([(t.hour >= 22 or t.hour < 6) for t in all_times]).mean()) if all_times else 0.0
        first_24h = float(sum((t - created_at).total_seconds() <= 86400 for t in all_times)) if not pd.isna(created_at) else 0.0
        first_7d = float(sum((t - created_at).total_seconds() <= 604800 for t in all_times)) if not pd.isna(created_at) else 0.0

        message_sent = float(row.get("messages_sent", 0) or 0)
        message_received = float(row.get("messages_received", 0) or 0)
        match_requests = float(row.get("match_requests_sent", 0) or 0)
        match_accepts = float(row.get("match_accepts", 0) or 0)

        device_type = str(row.get("device_type", "") or "").strip().lower()
        device_switch_count = 0.0
        dominant_device_ratio = 1.0 if device_type else 0.0
        unique_devices = 1.0 if device_type else 0.0

        login_ips = row.get("login_ip_list")
        if isinstance(login_ips, str):
            try:
                login_ips = json.loads(login_ips)
            except Exception:
                login_ips = []
        if not isinstance(login_ips, list):
            login_ips = []
        login_ips = [str(x) for x in login_ips if str(x).strip()]
        unique_locations = len(set(login_ips))
        location_switch_count = max(0, len(login_ips) - 1)
        location_switch_rate = _safe_div(location_switch_count, max(len(login_ips), 1))
        dominant_location_ratio = _safe_div(max((login_ips.count(ip) for ip in set(login_ips)), default=0), max(len(login_ips), 1))

        rows.append({
            "profile_id": row.get("profile_id"),
            "fraud_type": row.get("fraud_type"),
            "is_fraud": bool(row.get("is_fraud")),
            "m2b_target": int(_is_behavior_target(row)),
            "account_age_days": account_age_days,
            "total_events": float(len(all_times)),
            "events_per_day": per_day(len(all_times)),
            "login_count": float(len(login_times)),
            "logins_per_day": per_day(len(login_times)),
            "profile_edit_count": float(len(edit_times)),
            "edits_per_day": per_day(len(edit_times)),
            "upload_count": float(len(upload_times)),
            "uploads_per_day": per_day(len(upload_times)),
            "messages_sent": message_sent,
            "messages_received": message_received,
            "match_requests_sent": match_requests,
            "match_accepts": match_accepts,
            "message_send_rate": _safe_div(message_sent, max(account_age_days, 1.0)),
            "message_response_ratio": _safe_div(message_received, max(message_sent, 1.0)),
            "match_accept_rate": _safe_div(match_accepts, max(match_requests, 1.0)),
            "mean_inter_event_seconds": float(spacings.mean()) if len(spacings) else 0.0,
            "median_inter_event_seconds": float(np.median(spacings)) if len(spacings) else 0.0,
            "std_inter_event_seconds": float(spacings.std(ddof=0)) if len(spacings) else 0.0,
            "min_inter_event_seconds": float(spacings.min()) if len(spacings) else 0.0,
            "max_inter_event_seconds": float(spacings.max()) if len(spacings) else 0.0,
            "inter_event_cv": _safe_div(float(spacings.std(ddof=0)), float(spacings.mean())) if len(spacings) and spacings.mean() > 0 else 0.0,
            "burstiness": _burstiness(spacings),
            "night_activity_ratio": night_ratio,
            "weekend_activity_ratio": weekend_ratio,
            "activity_first_24h": _safe_div(first_24h, max(len(all_times), 1)),
            "activity_first_7d": _safe_div(first_7d, max(len(all_times), 1)),
            "unique_devices": unique_devices,
            "device_switch_count": device_switch_count,
            "device_switch_rate": _safe_div(device_switch_count, max(len(all_times), 1)),
            "dominant_device_ratio": dominant_device_ratio,
            "unique_locations": float(unique_locations),
            "location_switch_count": float(location_switch_count),
            "location_switch_rate": float(location_switch_rate),
            "dominant_location_ratio": float(dominant_location_ratio),
            "activity_density": _safe_div(len(all_times), max(account_age_days, 1.0)),
        })

    feat_df = pd.DataFrame(rows)
    return feat_df


def _is_behavior_target(row) -> bool:
    fraud_type = str(row.get("fraud_type") or "").strip()
    signals = str(row.get("injected_signals_str") or "")
    target_signals = any(sig in signals for sig in BEHAVIOR_SIGNAL_TARGETS)
    if fraud_type in {"template_bio", "functional", "financial_scam"} and target_signals:
        return True
    if fraud_type == "multi" and target_signals:
        return True
    return False


def preprocess_behavior_features(features: pd.DataFrame, fit_features: pd.DataFrame | None = None):
    metadata_columns = {"profile_id", "fraud_type", "is_fraud", "m2b_target"}
    cols = [
        c for c in features.columns
        if c not in metadata_columns and c not in EXCLUDED_MODEL_FEATURES
    ]
    def prepare(frame):
        out = frame[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        skew_cols = [c for c in out.columns if any(key in c for key in ("count", "events", "messages", "requests", "uploads", "logins"))]
        for col in skew_cols:
            out[col] = np.log1p(np.maximum(0.0, out[col]))
        return out

    fit_X = prepare(features if fit_features is None else fit_features)
    X = prepare(features)
    scaler = StandardScaler()
    scaler.fit(fit_X)
    Xs = scaler.transform(X)
    return Xs, scaler, cols
