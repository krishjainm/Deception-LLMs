#!/usr/bin/env python3
"""
Quick dataset validator for Deception-LLMs.

Usage:
  python schema_check.py --csv sample_data/normalized_poker_gpt4o.csv --expect_scenario poker
"""
import argparse, sys, pandas as pd

REQUIRED = ["statement","response","label","scenario"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--expect_scenario", default="poker")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"[FAIL] Missing columns: {missing}")
        sys.exit(1)

    bad_labels = df[~df["label"].isin([0,1])]
    if len(bad_labels):
        print(f"[FAIL] Non-binary labels detected: {bad_labels['label'].unique()[:10]} ...")
        sys.exit(2)

    bad_scen = df[df["scenario"] != args.expect_scenario]
    if len(bad_scen):
        print(f"[FAIL] Found rows with scenario!= '{args.expect_scenario}': {len(bad_scen)}")
        sys.exit(3)

    print("[OK] Columns present.")
    print("[OK] Labels are binary (0 truthful / 1 deceptive).")
    print(f"[OK] Scenario == '{args.expect_scenario}' for all rows.")
    print()
    print("Counts:")
    print(df["label"].value_counts().rename({0:"truthful",1:"deceptive"}))
    print()
    print("Sample rows:")
    print(df.sample(3, random_state=7)[REQUIRED].to_string(index=False))

if __name__ == "__main__":
    main()
