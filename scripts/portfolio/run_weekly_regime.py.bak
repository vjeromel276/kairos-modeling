#!/usr/bin/env python3
"""
run_weekly_regime.py

Weekly regime decision runner.

Responsibilities:
- Determine market_date (price clock)
- Determine decision_date (information clock)
- Read current regime
- Apply frozen throttle policy (YAML)
- Persist a single regime decision row

This script:
- NEVER builds portfolios
- NEVER runs backtests
- NEVER modifies upstream data
"""

import argparse
import sys
from datetime import datetime, date
import duckdb
import yaml
from pathlib import Path


DB_PATH = "data/kairos.duckdb"
THROTTLE_PATH = "scripts/portfolio/throttle_v1.yaml"

REGIME_ALIAS = {
    "normal_vol_neutral": "normal_vol_bull",
    "high_vol_neutral":   "high_vol_bull",
    "low_vol_neutral":    "low_vol_bull",
}

# -------------------------
# Helpers
# -------------------------

def max_date(con, table: str):
    try:
        return con.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
    except duckdb.CatalogException:
        return None


def resolve_last_market_date(con):
    row = con.execute("SELECT MAX(date) FROM sep_base").fetchone()
    return row[0]


def load_throttle(path: str):
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if "throttle" not in data:
        raise ValueError("Throttle YAML missing 'throttle' section")
    return data["throttle"], data.get("version", "unknown")


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Run weekly regime decision")
    args = parser.parse_args()

    # --- Connect DB ---
    try:
        con = duckdb.connect(DB_PATH)
    except Exception as e:
        print(f"ERROR: Could not connect to DuckDB: {e}")
        sys.exit(1)

    # --- Resolve clocks ---
    market_date = resolve_last_market_date(con)
    if market_date is None:
        print("ERROR: Could not resolve market date from sep_base")
        sys.exit(1)

    v7_date = max_date(con, "feat_composite_v7")
    if v7_date is None:
        print("ERROR: feat_composite_v7 not found")
        sys.exit(1)

    decision_date = min(market_date, v7_date)

    # --- Fetch regime ---
    regime_row = con.execute(
        """
        SELECT regime
        FROM regime_history_academic
        WHERE date = ?
        """,
        [market_date],
    ).fetchone()

    if regime_row is None:
        print(f"ERROR: No regime found for market_date={market_date}")
        sys.exit(1)

    raw_regime = regime_row[0]
    regime = REGIME_ALIAS.get(raw_regime, raw_regime)

    # --- Load throttle ---
    try:
        throttle_map, throttle_version = load_throttle(THROTTLE_PATH)
    except Exception as e:
        print(f"ERROR loading throttle YAML: {e}")
        sys.exit(1)

    if regime not in throttle_map:
        print(f"ERROR: Regime '{regime}' not found in throttle config")
        sys.exit(1)

    weights = throttle_map[regime]
    risk4_weight = weights.get("risk4_weight")
    hedge_weight = weights.get("hedge_weight")

    if risk4_weight is None or hedge_weight is None:
        print(f"ERROR: Invalid throttle entry for regime '{regime}'")
        sys.exit(1)

    if abs((risk4_weight + hedge_weight) - 1.0) > 1e-6:
        print("ERROR: risk4_weight + hedge_weight must equal 1.0")
        sys.exit(1)

    # --- Idempotency check ---
    existing = con.execute(
        """
        SELECT COUNT(*)
        FROM portfolio_regime_state
        WHERE date = ?
        """,
        [market_date],
    ).fetchone()[0]

    if existing > 0:
        print(f"INFO: Regime decision already exists for {market_date}, skipping.")
        sys.exit(0)

    # --- Persist decision ---
    con.execute(
        """
        INSERT INTO portfolio_regime_state (
            date,
            regime,
            risk4_weight,
            hedge_weight,
            notes,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [
            market_date,
            regime,
            risk4_weight,
            hedge_weight,
            f"throttle_version={throttle_version}; decision_date={decision_date}",
        ],
    )

    con.close()

    # --- Console summary ---
    print("\nWEEKLY REGIME DECISION")
    print("=====================")
    print(f"Market date:    {market_date}")
    print(f"Decision date:  {decision_date}")
    print(f"Regime:         {regime}")
    print(f"Risk4 weight:   {risk4_weight:.2f}")
    print(f"Hedge weight:   {hedge_weight:.2f}")
    print(f"Throttle ver.:  {throttle_version}")
    print("\n✓ Regime decision saved\n")


if __name__ == "__main__":
    main()
