#!/usr/bin/env python3
"""
run_weekly_hedge.py

Weekly hedge portfolio builder.

- Uses hedge_weight from portfolio_regime_state
- Builds beta-weighted short hedge
- Uses ONLY existing columns in feat_matrix_v2
"""

import sys
from datetime import datetime
import duckdb
import pandas as pd
import numpy as np

DB_PATH = "data/kairos.duckdb"
ADV_MIN = 2_000_000
VOL_EPS = 1e-6
TOP_K_HEDGE = 150   # number of names in hedge basket
BETA_CAP = 3.0   # standard; you can argue 2–4 later


def main():
    con = duckdb.connect(DB_PATH)

    # -------------------------
    # Resolve dates
    # -------------------------
    market_date = con.execute(
        "SELECT MAX(date) FROM sep_base"
    ).fetchone()[0]

    regime_row = con.execute("""
        SELECT date, hedge_weight
        FROM portfolio_regime_state
        WHERE date = ?
    """, [market_date]).fetchone()

    if regime_row is None:
        print(f"ERROR: No regime decision found for {market_date}")
        sys.exit(1)

    decision_date = con.execute(
        "SELECT MAX(date) FROM feat_composite_v7"
    ).fetchone()[0]

    hedge_weight = regime_row[1]

    if hedge_weight <= 0:
        print("INFO: hedge_weight = 0. No hedge required this week.")
        sys.exit(0)

    # -------------------------
    # Idempotency
    # -------------------------
    exists = con.execute("""
        SELECT COUNT(*)
        FROM portfolio_hedge_holdings
        WHERE decision_date = ?
    """, [decision_date]).fetchone()[0]

    if exists > 0:
        print(f"INFO: Hedge already exists for {decision_date}, skipping.")
        sys.exit(0)

    # -------------------------
    # Load hedge universe
    # -------------------------
    df = con.execute("""
        SELECT
            ticker,
            beta_252d,
            adv_20
        FROM feat_matrix_v2
        WHERE date = ?
          AND beta_252d IS NOT NULL
          AND adv_20 >= ?
    """, [decision_date, ADV_MIN]).fetchdf()

    if df.empty:
        print("ERROR: No eligible hedge universe")
        sys.exit(1)

    # -------------------------
    # Beta-weighted hedge
    # -------------------------
    df["beta_abs"] = df["beta_252d"].abs().clip(lower=VOL_EPS, upper=BETA_CAP)

    # Concentrate hedge to top-K by absolute beta
    df = df.sort_values("beta_abs", ascending=False).head(TOP_K_HEDGE).copy()

    df["raw_weight"] = df["beta_abs"] / df["beta_abs"].sum()

    # Apply hedge scale (negative weights)
    df["weight"] = -hedge_weight * df["raw_weight"]


    # -------------------------
    # Persist
    # -------------------------
    out = df[["ticker", "weight", "beta_252d", "adv_20"]].copy()
    out["decision_date"] = decision_date
    out["market_date"] = market_date
    out["created_at"] = datetime.utcnow()

    con.execute("""
        INSERT INTO portfolio_hedge_holdings (
            decision_date,
            market_date,
            ticker,
            weight,
            beta_252d,
            adv_20,
            created_at
        )
        SELECT
            decision_date,
            market_date,
            ticker,
            weight,
            beta_252d,
            adv_20,
            created_at
        FROM out
    """)

    con.close()

    # -------------------------
    # Console summary
    # -------------------------
    print("\nWEEKLY HEDGE PORTFOLIO")
    print("======================")
    print(f"Decision date: {decision_date}")
    print(f"Market date:   {market_date}")
    print(f"Hedge weight:  {hedge_weight:.2%}")
    print(f"Hedge names:   {len(out)}")
    print(f"Total hedge:   {out['weight'].sum():.2%}")
    print("\n✓ Hedge holdings saved\n")


if __name__ == "__main__":
    main()
