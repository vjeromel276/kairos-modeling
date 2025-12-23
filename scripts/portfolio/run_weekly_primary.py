#!/usr/bin/env python3
"""
run_weekly_primary.py

Weekly primary portfolio builder (long-only).

- Uses DECISION DATE (information clock)
- Executes on MARKET DATE (price clock)
- Reads alpha_composite_v7 from feat_matrix_v2
- Sizes by inverse vol_blend
- Writes labeled, auditable holdings
"""

import sys
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/kairos.duckdb"
TOP_N = 75
ADV_MIN = 2_000_000
MAX_WEIGHT = 0.03
VOL_EPS = 1e-6


def zscore_clip(s: pd.Series, clip: float = 3.0) -> pd.Series:
    mu = s.mean()
    sd = s.std()
    if sd == 0 or np.isnan(sd):
        return pd.Series(0.0, index=s.index)
    return ((s - mu) / sd).clip(-clip, clip)


def main():
    con = duckdb.connect(DB_PATH)

    # --- Resolve clocks ---
    market_date = con.execute(
        "SELECT MAX(date) FROM sep_base"
    ).fetchone()[0]

    decision_date = con.execute(
        "SELECT MAX(date) FROM feat_composite_v7"
    ).fetchone()[0]

    if decision_date is None:
        print("ERROR: No alpha_composite_v7 available")
        sys.exit(1)

    # --- Idempotency ---
    exists = con.execute("""
        SELECT COUNT(*) 
        FROM portfolio_primary_holdings
        WHERE decision_date = ?
    """, [decision_date]).fetchone()[0]

    if exists > 0:
        print(f"INFO: Holdings already exist for {decision_date}, skipping.")
        sys.exit(0)

    # --- Load universe slice ---
    df = con.execute("""
        SELECT
            ticker,
            alpha_composite_v7 AS alpha,
            vol_blend,
            adv_20
        FROM feat_matrix_v2
        WHERE date = ?
          AND alpha_composite_v7 IS NOT NULL
          AND adv_20 >= ?
    """, [decision_date, ADV_MIN]).fetchdf()

    if df.empty:
        print("ERROR: No eligible universe rows")
        sys.exit(1)

    # --- Rank ---
    df["alpha_z"] = zscore_clip(df["alpha"])
    df = df.sort_values("alpha_z", ascending=False).head(TOP_N)

    # --- Risk sizing ---
    df["inv_vol"] = 1.0 / df["vol_blend"].clip(lower=VOL_EPS)
    df["raw_weight"] = df["inv_vol"] * df["alpha_z"].clip(lower=0.0)

    # Drop zero-weight names
    df = df[df["raw_weight"] > 0].copy()

    # Normalize + cap
    df["weight"] = df["raw_weight"] / df["raw_weight"].sum()
    df["weight"] = df["weight"].clip(upper=MAX_WEIGHT)
    df["weight"] = df["weight"] / df["weight"].sum()

    # Rank after final selection
    df = df.sort_values("alpha_z", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # --- Build output ---
    out = df[[
        "ticker",
        "weight",
        "alpha",
        "alpha_z",
        "rank",
        "vol_blend",
        "adv_20"
    ]].copy()

    out["decision_date"] = decision_date
    out["market_date"] = market_date
    out["created_at"] = datetime.utcnow()

    # --- Insert ---
    con.execute("""
        INSERT INTO portfolio_primary_holdings (
            decision_date,
            market_date,
            ticker,
            weight,
            alpha,
            alpha_z,
            rank,
            vol_blend,
            adv_20,
            created_at
        )
        SELECT
            decision_date,
            market_date,
            ticker,
            weight,
            alpha,
            alpha_z,
            rank,
            vol_blend,
            adv_20,
            created_at
        FROM out
    """)

    con.close()

    # --- Console summary ---
    print("\nWEEKLY PRIMARY PORTFOLIO")
    print("========================")
    print(f"Decision date: {decision_date}")
    print(f"Market date:   {market_date}")
    print(f"Holdings:      {len(out)}")
    print(f"Top weight:    {out['weight'].max():.2%}")
    print(f"Total weight:  {out['weight'].sum():.2%}")
    print("\n✓ Primary holdings saved\n")


if __name__ == "__main__":
    main()
