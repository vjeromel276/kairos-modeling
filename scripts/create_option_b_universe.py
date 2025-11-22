#!/usr/bin/env python3
"""
create_option_b_universe.py  (patched to use sep_base)

Builds an expanded equity universe ("Option B") using the incremental
SHARADAR table `sep_base` instead of `sep`, so daily updates flow directly
into the universe.

Universe rules (Option B):
- Category = 'Domestic Common Stock'
- Exchange IN ('NYSE','NASDAQ','AMEX')
- Market cap bucket IN ('3 - Small','4 - Mid','5 - Large','6 - Mega')
- Latest close >= --min-price
- 60-day ADV >= --min-adv
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
# Core builder
# ---------------------------------------------------------------------
def build_option_b_universe(con: duckdb.DuckDBPyConnection, min_adv: float, min_price: float):

    # Ensure required tables exist
    tables = con.execute("SHOW TABLES").fetchdf()["name"].str.lower().tolist()
    if "sep_base" not in tables:
        raise RuntimeError("❌ sep_base table not found. Run merge_daily_download_duck.py first.")

    if "tickers" not in tables:
        raise RuntimeError("❌ tickers table not found in DuckDB.")

    logger.info("Confirmed presence of sep_base + tickers")

    # 1) Latest date
    max_date = con.execute("SELECT MAX(date) FROM sep_base").fetchone()[0]
    logger.info(f"Latest date in sep_base: {max_date}")

    # 2) Compute 60-day average ADV
    logger.info("Computing avg dollar volume over ~60 trading days...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE __recent_dollar_vol AS
        WITH maxd AS (SELECT MAX(date) AS max_date FROM sep_base)
        SELECT
            s.ticker,
            s.date,
            s.close * s.volume AS dollar_volume
        FROM sep_base s
        JOIN maxd
          ON s.date >= maxd.max_date - INTERVAL 90 DAY
         AND s.date <= maxd.max_date
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE __adv_60d AS
        SELECT ticker, AVG(dollar_volume) AS adv_60d
        FROM __recent_dollar_vol
        GROUP BY ticker
    """)

    # 3) Latest close per ticker
    logger.info("Computing latest close per ticker...")
    con.execute("""
        CREATE OR REPLACE TEMP TABLE __latest_px AS
        WITH last_dates AS (
            SELECT ticker, MAX(date) AS max_date
            FROM sep_base
            GROUP BY ticker
        )
        SELECT s.ticker, s.close AS last_close
        FROM sep_base s
        JOIN last_dates d
          ON s.ticker = d.ticker
         AND s.date = d.max_date
    """)

    # 4) Ticker-level universe
    logger.info("Building ticker-level universe...")
    universe = con.execute(
        """
        SELECT
            t.ticker,
            t.name,
            t.exchange,
            t.category,
            t.scalemarketcap,
            t.isdelisted,
            adv.adv_60d,
            px.last_close
        FROM tickers t
        JOIN __adv_60d adv ON t.ticker = adv.ticker
        JOIN __latest_px px ON t.ticker = px.ticker
        WHERE t.category = 'Domestic Common Stock'
          AND t.exchange IN ('NYSE','NASDAQ','AMEX')
          AND t.scalemarketcap IN ('3 - Small','4 - Mid','5 - Large','6 - Mega')
          AND adv.adv_60d >= ?
          AND px.last_close >= ?
          AND (
                t.isdelisted IS NULL OR
                CAST(t.isdelisted AS VARCHAR) IN ('N','0','False','FALSE')
          )
        """,
        (min_adv, min_price)
    ).fetchdf()

    logger.info(f"Universe size: {len(universe)} tickers")

    # 5) Build sep_base_expanded (all history)
    logger.info("Building sep_base_expanded with full history...")
    con.execute("DROP TABLE IF EXISTS sep_base_expanded")

    con.execute(f"""
        CREATE TABLE sep_base_expanded AS
        SELECT s.*
        FROM sep_base AS s
        WHERE s.ticker IN (
            SELECT DISTINCT t.ticker
            FROM tickers AS t
            JOIN __adv_60d AS adv
              ON t.ticker = adv.ticker
            JOIN __latest_px AS px
              ON t.ticker = px.ticker
            WHERE t.category = 'Domestic Common Stock'
              AND t.exchange IN ('NYSE','NASDAQ','AMEX')
              AND t.scalemarketcap IN ('3 - Small','4 - Mid','5 - Large','6 - Mega')
              AND adv.adv_60d >= {min_adv}
              AND px.last_close >= {min_price}
              AND (
                    t.isdelisted IS NULL OR
                    CAST(t.isdelisted AS VARCHAR) IN ('N','0','False','FALSE')
              )
        )
    """)

    stats = con.execute("""
        SELECT
            COUNT(*) AS rows,
            COUNT(DISTINCT ticker) AS tickers,
            MIN(date) AS start_date,
            MAX(date) AS end_date
        FROM sep_base_expanded
    """).fetchdf()

    logger.info(
        "sep_base_expanded → %d rows, %d tickers, [%s → %s]",
        stats.loc[0,"rows"],
        stats.loc[0,"tickers"],
        stats.loc[0,"start_date"],
        stats.loc[0,"end_date"],
    )

    return universe


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/kairos.duckdb")
    parser.add_argument("--min-adv", type=float, default=500000)
    parser.add_argument("--min-price", type=float, default=2.0)
    parser.add_argument("--universe-csv", type=str, default="")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    universe = build_option_b_universe(con, args.min_adv, args.min_price)

    if args.universe_csv:
        os.makedirs(os.path.dirname(args.universe_csv), exist_ok=True)
        universe.to_csv(args.universe_csv, index=False)
        logger.info(f"Saved universe CSV → {args.universe_csv}")

    con.close()


if __name__ == "__main__":
    main()
