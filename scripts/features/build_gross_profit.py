#!/usr/bin/env python3
"""
build_gross_profit.py
=====================
Build feat_gross_profit table: Novy-Marx (2013) gross profitability factor.

Formula:
    gross_profitability = gp / assets
    where gp = gross profit (revenue - COGS), assets = total assets
    Source: Sharadar SF1, ARQ dimension (as-reported quarterly)

Method:
    1. Extract quarterly gross_profitability from sf1 (ARQ)
    2. Forward-fill to daily trading dates using sep_base_academic
       (universe tickers only — same scope as all other feat_* tables)
    3. Write feat_gross_profit with explicit CAST(date AS DATE)

Output table: feat_gross_profit
    Columns: ticker, date (DATE), gross_profitability (DOUBLE)

Usage:
    python scripts/features/build_gross_profit.py --db data/kairos.duckdb

Pipeline integration:
    Add to run_pipeline.py Phase 3 (Fundamental Factors), after
    rebuild_feat_fundamental.py
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MIN_DATE = "2000-01-01"
BATCH_SIZE = 500_000


def extract_quarterly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract gross_profitability from sf1 ARQ dimension.

    Uses NULLIF to guard against division by zero.
    Negative assets are excluded — they indicate data errors in SF1.
    Winsorize at [1st, 99th] percentile to remove extreme outliers
    before forward-filling (same approach as rebuild_feat_fundamental.py).
    """
    logger.info("Extracting SF1 quarterly data (ARQ dimension)...")

    df = con.execute("""
        SELECT
            ticker,
            CAST(datekey AS DATE) AS date,
            CAST(gp AS DOUBLE) / NULLIF(CAST(assets AS DOUBLE), 0)
                AS gross_profitability
        FROM sf1
        WHERE dimension = 'ARQ'
          AND assets > 0
          AND gp IS NOT NULL
          AND datekey >= '2000-01-01'
        ORDER BY ticker, date
    """).fetchdf()

    logger.info(f"  Raw quarterly rows: {len(df):,}")
    logger.info(f"  Tickers: {df['ticker'].nunique():,}")

    # Winsorise at 1st / 99th percentile — same pattern as value/quality builds
    p01 = df["gross_profitability"].quantile(0.01)
    p99 = df["gross_profitability"].quantile(0.99)
    df["gross_profitability"] = df["gross_profitability"].clip(p01, p99)
    logger.info(f"  Winsorised range: [{p01:.4f}, {p99:.4f}]")

    # Drop rows where the ratio is still NaN after the query
    before = len(df)
    df = df.dropna(subset=["gross_profitability"])
    if before != len(df):
        logger.info(f"  Dropped {before - len(df):,} NaN rows after winsorise")

    return df


def get_universe_daily_grid(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Get all (ticker, date) pairs from sep_base_academic.
    This is the same universe grid used by every other feat_* table.
    """
    logger.info("Loading daily universe grid from sep_base_academic...")

    grid = con.execute("""
        SELECT DISTINCT
            ticker,
            CAST(date AS DATE) AS date
        FROM sep_base_academic
        WHERE date >= '2000-01-01'
        ORDER BY ticker, date
    """).fetchdf()

    grid["date"] = pd.to_datetime(grid["date"])
    logger.info(f"  Grid rows: {len(grid):,}")
    logger.info(f"  Universe tickers: {grid['ticker'].nunique():,}")
    return grid


def forward_fill_and_save(
    con: duckdb.DuckDBPyConnection,
    quarterly: pd.DataFrame,
) -> None:
    """
    Forward-fill quarterly gross_profitability to daily frequency and
    write directly to feat_gross_profit.

    Uses DuckDB's native ASOF JOIN, which handles per-ticker forward-fill
    correctly without requiring a global date sort. This is faster than
    pandas merge_asof for large grids and avoids the 'left keys must be
    sorted' error that occurs when date resets per ticker.

    ASOF JOIN semantics: for each (ticker, date) in the grid, find the
    most recent quarterly row where quarterly.date <= grid.date — i.e.,
    carry the last known quarterly value forward.
    """
    logger.info("Forward-filling quarterly data to daily grid (DuckDB ASOF JOIN)...")

    # Register the quarterly DataFrame as a DuckDB view
    con.register("quarterly_gp", quarterly)

    # Build feat_gross_profit directly in DuckDB:
    # 1. ASOF LEFT JOIN fills each daily date with the most recent quarterly value
    # 2. CAST(grid.date AS DATE) ensures correct column type
    # 3. WHERE gross_profitability IS NOT NULL drops grid rows with no quarterly history
    con.execute("DROP TABLE IF EXISTS feat_gross_profit")

    con.execute("""
        CREATE TABLE feat_gross_profit AS
        SELECT
            grid.ticker,
            CAST(grid.date AS DATE) AS date,
            q.gross_profitability
        FROM (
            SELECT
                ticker,
                CAST(date AS DATE) AS date
            FROM sep_base_academic
            WHERE date >= '2000-01-01'
        ) AS grid
        ASOF LEFT JOIN (
            SELECT
                ticker,
                CAST(date AS DATE) AS date,
                gross_profitability
            FROM quarterly_gp
        ) AS q
            ON grid.ticker = q.ticker
            AND grid.date >= q.date
        WHERE q.gross_profitability IS NOT NULL
    """)

    count = con.execute(
        "SELECT COUNT(*) FROM feat_gross_profit"
    ).fetchone()[0]
    logger.info(f"  Created feat_gross_profit with {count:,} rows")

    # Coverage check against the full grid
    grid_count = con.execute(
        "SELECT COUNT(*) FROM sep_base_academic WHERE date >= '2000-01-01'"
    ).fetchone()[0]
    coverage = count / grid_count * 100
    logger.info(f"  Coverage: {count:,} / {grid_count:,} ({coverage:.1f}%)")

    # Index for join performance — same pattern as other feat_* tables
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_gp_ticker_date
        ON feat_gross_profit(ticker, date)
    """)
    logger.info("  Index created")


def validate(con: duckdb.DuckDBPyConnection) -> None:
    """Print summary statistics and spot-check the output table."""

    logger.info("\nValidation:")

    stats = con.execute("""
        SELECT
            MIN(date)                        AS min_date,
            MAX(date)                        AS max_date,
            COUNT(DISTINCT ticker)           AS n_tickers,
            COUNT(*)                         AS n_rows,
            AVG(gross_profitability)         AS mean_gp,
            STDDEV(gross_profitability)      AS std_gp,
            MIN(gross_profitability)         AS min_gp,
            MAX(gross_profitability)         AS max_gp
        FROM feat_gross_profit
    """).fetchdf()

    logger.info(f"  Date range:  {stats['min_date'].iloc[0]} to {stats['max_date'].iloc[0]}")
    logger.info(f"  Tickers:     {stats['n_tickers'].iloc[0]:,}")
    logger.info(f"  Total rows:  {stats['n_rows'].iloc[0]:,}")
    logger.info(f"  Mean GP:     {stats['mean_gp'].iloc[0]:.4f}")
    logger.info(f"  Std GP:      {stats['std_gp'].iloc[0]:.4f}")
    logger.info(f"  Range:       [{stats['min_gp'].iloc[0]:.4f}, {stats['max_gp'].iloc[0]:.4f}]")

    # Spot-check date type — must be DATE not TIMESTAMP
    dtype_check = con.execute("""
        SELECT typeof(date) AS date_type
        FROM feat_gross_profit
        LIMIT 1
    """).fetchone()
    logger.info(f"  date column type: {dtype_check[0]}")

    if dtype_check[0].upper() != "DATE":
        logger.warning(
            "  *** date column is not DATE type — joins may silently fail ***"
        )

    # Cross-check coverage against sep_base_academic universe
    coverage = con.execute("""
        SELECT
            COUNT(DISTINCT gp.ticker) AS gp_tickers,
            COUNT(DISTINCT sep.ticker) AS universe_tickers,
            ROUND(
                100.0 * COUNT(DISTINCT gp.ticker)
                / NULLIF(COUNT(DISTINCT sep.ticker), 0),
                1
            ) AS pct_covered
        FROM sep_base_academic sep
        LEFT JOIN feat_gross_profit gp
            ON sep.ticker = gp.ticker
    """).fetchone()
    logger.info(
        f"  Universe coverage: {coverage[0]:,} / {coverage[1]:,} tickers "
        f"({coverage[2]:.1f}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build feat_gross_profit from SF1 ARQ data"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("BUILD FEAT_GROSS_PROFIT (Novy-Marx 2013)")
    logger.info("=" * 70)
    logger.info(f"Database: {args.db}")
    logger.info(f"Formula:  gross_profitability = gp / assets (SF1 ARQ)")

    con = duckdb.connect(args.db)

    try:
        # Step 1: quarterly SF1 data
        quarterly = extract_quarterly(con)

        # Step 2: forward-fill to daily and write directly to DB
        forward_fill_and_save(con, quarterly)

        # Step 3: validate
        validate(con)

        logger.info("\n" + "=" * 70)
        logger.info("DONE - feat_gross_profit built successfully")
        logger.info("=" * 70)
        logger.info("Next step:")
        logger.info(
            "  Verify feat_trend has price_vs_sma_21, sma_21_slope, macd_hist"
        )
        logger.info("  then run train_xgb_alpha_v3.py in the research container")

    finally:
        con.close()


if __name__ == "__main__":
    main()