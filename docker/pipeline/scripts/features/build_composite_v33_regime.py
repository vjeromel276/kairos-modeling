#!/usr/bin/env python3
"""
build_composite_v33_regime.py

Composite v33 (Regime-Aware Horizon Blend)

Uses 6-regime mapping:
    low_vol_bull
    low_vol_bear
    normal_vol_bull
    normal_vol_bear
    high_vol_bull
    high_vol_bear

Weights (CS, CL2, SM):

    low_vol_bull:      (0.50, 0.25, 0.25)
    low_vol_bear:      (0.25, 0.50, 0.25)
    normal_vol_bull:   (0.40, 0.30, 0.30)
    normal_vol_bear:   (0.25, 0.45, 0.30)
    high_vol_bull:     (0.25, 0.50, 0.25)
    high_vol_bear:     (0.20, 0.55, 0.25)

Output table:
    feat_composite_v33_regime
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================
# Regime → weight mapping (CS, CL2, SM)
# =============================================================

REGIME_WEIGHTS = {
    # Low volatility
    "low_vol_bull":      (0.50,  0.25,  0.25),
    "low_vol_neutral":   (0.375, 0.375, 0.25),   # NEW
    "low_vol_bear":      (0.25,  0.50,  0.25),
    
    # Normal volatility
    "normal_vol_bull":   (0.40,  0.30,  0.30),
    "normal_vol_neutral":(0.325, 0.375, 0.30),   # NEW
    "normal_vol_bear":   (0.25,  0.45,  0.30),
    
    # High volatility
    "high_vol_bull":     (0.25,  0.50,  0.25),
    "high_vol_neutral":  (0.225, 0.525, 0.25),   # NEW
    "high_vol_bear":     (0.20,  0.55,  0.25),
}


def load_component_tables(con):
    """Load CS, CL2, SM, and regimes."""

    logger.info("Loading CS (alpha_composite_eq)...")
    cs = con.execute("""
        SELECT ticker, date, alpha_composite_eq AS alpha_cs
        FROM feat_composite_academic
    """).fetchdf()
    logger.info(f"CS rows: {len(cs):,}")

    logger.info("Loading CL2 (alpha_CL_v2)...")
    cl2 = con.execute("""
        SELECT ticker, date, alpha_CL_v2
        FROM feat_composite_long_v2
    """).fetchdf()
    logger.info(f"CL2 rows: {len(cl2):,}")

    logger.info("Loading smoothed alpha (alpha_smoothed)...")
    sm = con.execute("""
        SELECT ticker, date, alpha_smoothed
        FROM feat_alpha_smoothed_v31
    """).fetchdf()
    logger.info(f"SM rows: {len(sm):,}")

    logger.info("Loading regimes...")
    regimes = con.execute("""
        SELECT date, regime
        FROM regime_history_academic
    """).fetchdf()
    logger.info(f"Regime rows: {len(regimes):,}")

    return cs, cl2, sm, regimes


def build_v33(con):
    """Build regime-aware composite factor v33."""

    cs, cl2, sm, regimes = load_component_tables(con)

    # Merge CS (primary)
    logger.info("Merging CS + CL2 + SM...")
    df = cs.merge(cl2, on=["ticker", "date"], how="left")
    df = df.merge(sm, on=["ticker", "date"], how="left")

    # Merge regimes by date
    df = df.merge(regimes, on="date", how="left")

    logger.info(f"Joined rows (before dropna): {len(df):,}")

    # Drop rows without a regime label
    df = df.dropna(subset=["regime"])
    logger.info(f"Rows after regime merge: {len(df):,}")

    # Apply regime weights
    def apply_regime_blend(row):
        regime = row["regime"]
        if regime not in REGIME_WEIGHTS:
            return np.nan

        w_cs, w_cl2, w_sm = REGIME_WEIGHTS[regime]

        cs_val = row["alpha_cs"]
        cl2_val = row["alpha_CL_v2"]
        sm_val = row["alpha_smoothed"]

        if pd.isna(cs_val) and pd.isna(cl2_val) and pd.isna(sm_val):
            return np.nan

        cs_val = 0.0 if pd.isna(cs_val) else cs_val
        cl2_val = 0.0 if pd.isna(cl2_val) else cl2_val
        sm_val = 0.0 if pd.isna(sm_val) else sm_val

        return (
            w_cs  * cs_val +
            w_cl2 * cl2_val +
            w_sm  * sm_val
        )

    logger.info("Computing regime-aware composite...")
    df["alpha_composite_v33_regime"] = df.apply(apply_regime_blend, axis=1)

    # Enforce uniqueness
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"])
    after = len(df)
    logger.info(f"Deduped {before - after:,} rows. Final: {after:,}")

    logger.info("Saving feat_composite_v33_regime...")
    con.execute("DROP TABLE IF EXISTS feat_composite_v33_regime")
    con.register("df_v33", df)
    con.execute("CREATE TABLE feat_composite_v33_regime AS SELECT * FROM df_v33")

    logger.info("feat_composite_v33_regime created successfully.")


def main():
    parser = argparse.ArgumentParser(description="Build Regime-Aware Composite v33")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_v33(con)
    con.close()


if __name__ == "__main__":
    main()
