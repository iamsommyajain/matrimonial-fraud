"""
regenerate_reports.py

Rebuilds output/reports.csv from an already-generated profiles.csv, without
touching profiles.csv, profiles.parquet, the splits, or the interaction
graph. Use this after changing report_generator.py's logic (report
probability, volume, weighting) so you don't have to re-run the full
50,000-profile generation just to get new report data.

Run from the project root:
    python -m dataset.regenerate_reports --profiles ./output/profiles.csv --output_dir ./output
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import pandas as pd

if __package__:
    from .report_generator import generate_all_reports
else:
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dataset.report_generator import generate_all_reports


def _load_profile_records(profiles_path: str) -> list[dict]:
    df = pd.read_csv(profiles_path, usecols=["profile_id", "is_fraud", "fraud_type", "created_at"])
    df["is_fraud"] = df["is_fraud"].map(
        lambda v: v if isinstance(v, (bool,)) else str(v).strip().lower() in {"true", "1"}
    )
    return df.to_dict(orient="records")


def main():
    parser = argparse.ArgumentParser(description="Regenerate output/reports.csv from an existing profiles.csv")
    parser.add_argument("--profiles", default="./output/profiles.csv")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading profiles from {args.profiles}...")
    records = _load_profile_records(args.profiles)
    print(f"  {len(records):,} profiles loaded")

    print("Generating community reports...")
    reports = generate_all_reports(records)
    reports_df = pd.DataFrame(reports)

    out_path = os.path.join(args.output_dir, "reports.csv")
    reports_df.to_csv(out_path, index=False)

    print(f"  Reports generated: {len(reports_df):,}")
    print(f"  Fraud reports:     {int(reports_df['reported_is_fraud'].sum()):,}")
    print(f"  Noise reports:     {int((~reports_df['reported_is_fraud']).sum()):,}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
