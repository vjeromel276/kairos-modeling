#!/usr/bin/env python3
"""
create_academic_base.py

Purpose:
    Build the canonical, academically clean base dataset for all factor,
    regime, and target computation.

    This script:
        1. Reads the Option B ticker universe CSV (filtered tickers only)
        2. Loads full historical price data from sep_base (incremental table)
        3. Filters rows to universe tickers
        4. Sorts by (ticker, date)
        5. Creates optimized table: sep_base_academic
        6. Runs PRAGMA optimize for compression & performance

This is the successor to the "sep_base_expanded" portion of
create_option_b_universe.py – universe filtering stays in that script,
but the heavy lifting of building the 10M-row academic table moves here.

Usage:
    python scripts/create_academic_base.py \
        --db data/kairos.duckdb \
        --universe scripts/sep_dataset/feature_sets/option_b_universe.csv
"""

import argparse
import logging
import os
import duckdb
import pandas as pd

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Build sep_base_academic
# ---------------------------------------------------------------------
def build_academic_base(con: duckdb.DuckDBPyConnection, universe_csv: str):

    if not os.path.exists(universe_csv):
        raise FileNotFoundError(
            f"Universe CSV not found: {universe_csv}"
        )

    logger.info(f"Loading universe CSV: {universe_csv}")
    universe_df = pd.read_csv(universe_csv)
    universe_tickers = universe_df["ticker"].unique().tolist()
    logger.info(f"Universe contains {len(universe_tickers):,} tickers")

    # Ensure sep_base exists
    tables = con.execute("SHOW TABLES").fetchdf()["name"].str.lower().tolist()
    if "sep_base" not in tables:
        raise RuntimeError("❌ sep_base table not found. Run merge_daily_download_duck.py first.")

    # Drop old table
    logger.info("Dropping old sep_base_academic (if exists)...")
    con.execute("DROP TABLE IF EXISTS sep_base_academic")

    # Build new table (sorted & filtered)
    logger.info("Building sep_base_academic (this may take ~30–80 seconds)...")

    # Register tickers as temp table for fast filtering
    con.register("universe_df", pd.DataFrame({"ticker": universe_tickers}))

    con.execute("""
        CREATE TABLE sep_base_academic AS
        SELECT
            s.*
        FROM sep_base AS s
        INNER JOIN universe_df u
            ON s.ticker = u.ticker
        ORDER BY s.ticker, s.date
    """)

    # Optimization for compression & index building
    logger.info("Running PRAGMA optimize...")
        # Optimization step (skip if unsupported)
    try:
        logger.info("Running PRAGMA optimize...")
        con.execute("PRAGMA optimize")
    except Exception:
        logger.info("Skipping PRAGMA optimize — unsupported on this DuckDB version.")


    # Summary
    stats = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM sep_base_academic
    """).fetchdf()

    logger.info("sep_base_academic → %d rows, %d tickers, [%s → %s]",
        stats.loc[0, "rows"],
        stats.loc[0, "tickers"],
        stats.loc[0, "start_date"],
        stats.loc[0, "end_date"],
    )

    logger.info("🎉 Academic base table created successfully!")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Build sep_base_academic (Option B universe).")
    parser.add_argument("--db", default="data/kairos.duckdb")
    parser.add_argument("--universe", required=True,
                        help="Path to the ticker universe CSV from create_option_b_universe.py")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Connecting to DuckDB: {args.db}")
    con = duckdb.connect(args.db)

    build_academic_base(con, args.universe)

    con.close()


if __name__ == "__main__":
    main()
