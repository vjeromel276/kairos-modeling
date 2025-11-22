#!/usr/bin/env python3
"""
regime_detector_academic.py

A CLEAN academic-only regime detection system using:
    - sep_base_academic (optimized price dataset)
    - Option B universe
    - volatility, trend, dispersion
    
Creates:
    DuckDB table: regime_history_academic

Usage:
    python scripts/regime_detector_academic.py --db data/kairos.duckdb
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

# Logging ---------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Compute volatility, trend, dispersion (academic style)
# ----------------------------------------------------------------------
def compute_regime_data(con):
    
    logger.info("Pulling daily close data from sep_base_academic...")
    
    df = con.execute("""
        SELECT
            date,
            ticker,
            close
        FROM sep_base_academic
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows of price data.")

    # ------------------------------------------------------------------
    # 1) Pivot into wide format (date × ticker) for fast cross-sectional ops
    # ------------------------------------------------------------------
    logger.info("Pivoting into wide daily close matrix...")
    
    prices = df.pivot(index="date", columns="ticker", values="close")
    prices = prices.sort_index()
    
    # forward-fill gaps (common in Sharadar small caps)
    prices = prices.ffill()

    # ------------------------------------------------------------------
    # 2) Daily returns matrix
    # ------------------------------------------------------------------
    logger.info("Computing daily returns...")
    rets = prices.pct_change()

    # ------------------------------------------------------------------
    # 3) Volatility regime: 21-day rolling market volatility
    # ------------------------------------------------------------------
    logger.info("Computing 21d volatility regime...")
    
    # Market volatility = cross-sectional median std dev
    vol_21 = rets.rolling(21).std().median(axis=1)

    # Quantile cuts
    vol_low = vol_21.quantile(0.33)
    vol_high = vol_21.quantile(0.66)

    def vol_bucket(v):
        if v <= vol_low: return "low_vol"
        if v >= vol_high: return "high_vol"
        return "normal_vol"

    vol_regime = vol_21.apply(vol_bucket)

    # ------------------------------------------------------------------
    # 4) Trend regime: 60-day SPY-style trend proxy = median return
    # ------------------------------------------------------------------
    logger.info("Computing 60d trend regime...")

    trend_60 = prices.pct_change(60).median(axis=1)

    trend_up = trend_60.quantile(0.66)
    trend_down = trend_60.quantile(0.33)

    def trend_bucket(x):
        if x <= trend_down: return "bear"
        if x >= trend_up: return "bull"
        return "neutral"

    trend_regime = trend_60.apply(trend_bucket)

    # ------------------------------------------------------------------
    # 5) Cross-sectional dispersion: stddev of daily returns
    # ------------------------------------------------------------------
    logger.info("Computing dispersion regime...")

    dispersion = rets.std(axis=1)
    disp_med = dispersion.median()

    disp_regime = dispersion.apply(
        lambda x: "high_dispersion" if x >= disp_med else "low_dispersion"
    )

    # ------------------------------------------------------------------
    # Combine regimes into final label
    # ------------------------------------------------------------------
    logger.info("Combining into final regime labels...")

    final_regime = (
        vol_regime + "_" +
        trend_regime
    )

    # ------------------------------------------------------------------
    # Build final DataFrame
    # ------------------------------------------------------------------
    out = pd.DataFrame({
        "date": prices.index,
        "vol_regime": vol_regime.values,
        "trend_regime": trend_regime.values,
        "dispersion_regime": disp_regime.values,
        "regime": final_regime.values
    }).dropna()

    logger.info(f"Final regime dataframe: {len(out):,} rows.")

    return out


# ----------------------------------------------------------------------
# Save regime table to DuckDB
# ----------------------------------------------------------------------
def save_regime_table(con, df):
    logger.info("Dropping old regime_history_academic table (if exists)...")
    con.execute("DROP TABLE IF EXISTS regime_history_academic")
    
    logger.info("Creating new regime_history_academic table...")
    con.register("regime_df", df)
    con.execute("""
        CREATE TABLE regime_history_academic AS
        SELECT * FROM regime_df
    """)
    logger.info("Saved regime_history_academic.")


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
def show_diagnostics(df):
    logger.info("\n================ REGIME DIAGNOSTICS ================\n")

    print("Regime Distribution:")
    print(df["regime"].value_counts(normalize=True).round(3), "\n")

    print("Vol Regime Distribution:")
    print(df["vol_regime"].value_counts(normalize=True).round(3), "\n")

    print("Trend Regime Distribution:")
    print(df["trend_regime"].value_counts(normalize=True).round(3), "\n")

    print("Dispersion Regime Distribution:")
    print(df["dispersion_regime"].value_counts(normalize=True).round(3), "\n")

    # Persistence
    print("Regime Persistence (avg days):")
    persistence = (
        df["regime"]
        .ne(df["regime"].shift())
        .cumsum()
        .groupby(df["regime"])
        .size()
        / df["regime"].value_counts()
    )
    print(persistence.round(2), "\n")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    regime_df = compute_regime_data(con)
    save_regime_table(con, regime_df)

    if not args.no_diagnostics:
        show_diagnostics(regime_df)

    con.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
