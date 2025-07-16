#!/usr/bin/env python3
"""
filter_common_duck.py

DuckDB-native pipeline to filter and export a universe of liquid U.S. common stocks.

This script applies the following filters:
  1. Category: Domestic Common Stock (case-insensitive)
  2. Not delisted
  3. Minimum 1-month average daily volume (ADV)
  4. Minimum trading days since start date
  5. Optional price floor to exclude penny stocks

Outputs:
  - Parquet file of filtered SHARADAR SEP data
  - CSV file of the ticker universe
  - DuckDB table for the filtered universe, named mid_cap_<YYYY_MM_DD>

Usage:
    python scripts/filter_common_duck.py \
        --db data/kairos.duckdb \
        --min-adv 2000000 \
        --min-days 252 \
        --start-date 1998-01-01 \
        --min-price 2.00 \
        --bucket midcap_and_up \
        --output-dir scripts/sep_dataset/feature_sets
"""
import argparse
import os
import logging
from datetime import datetime

import duckdb
import pandas as pd
from pandas.tseries.offsets import BDay
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def validate_date(date_str: str) -> pd.Timestamp:
    """
    Validate and parse a date string in YYYY-MM-DD format.
    Returns a pandas.Timestamp.
    """
    return pd.to_datetime(date_str, format="%Y-%m-%d")

def get_candidate_tickers(conn: duckdb.DuckDBPyConnection, min_adv: int) -> pd.DataFrame:
    """
    Fetch tickers with category common stock, not delisted, and ADV >= min_adv.
    """
    query = f"""
    SELECT t.ticker
    FROM tickers t
    JOIN (
        SELECT ticker, MAX(date) AS latest_date
        FROM metrics
        GROUP BY ticker
    ) ml ON t.ticker=ml.ticker
    JOIN metrics m ON m.ticker=ml.ticker AND m.date=ml.latest_date
    WHERE LOWER(t.category) LIKE '%common stock%'
      AND COALESCE(LOWER(t.isdelisted), 'false') IN ('false','0','n','no')
      AND m.volumeavg1m >= {min_adv}
    """
    logger.info("Filtering common stocks with ADV >= %d", min_adv)
    df = conn.execute(query).fetchdf()
    logger.info("Found %d candidate tickers", len(df))
    return df

def ensure_min_trading_days(
    conn: duckdb.DuckDBPyConnection,
    tickers: pd.Series,
    min_days: int,
    start_date: pd.Timestamp
) -> pd.Index:
    """
    Ensure each ticker has >= min_days since start_date.
    """
    logger.info("Checking trading-day history >= %d since %s", min_days, start_date.date())
    valid = []
    for t in tqdm(tickers, desc="Checking history"):
        q = f"""
        SELECT date FROM sep_base
        WHERE ticker='{t}' AND date>=DATE'{start_date.date()}'
        ORDER BY date
        """
        df = conn.execute(q).fetchdf()
        if len(df)<min_days:
            continue
        dates = pd.to_datetime(df['date']).dt.normalize().drop_duplicates()
        full = pd.date_range(dates.min(), dates.max(), freq=BDay())
        if dates.isin(full).sum()>=min_days:
            valid.append(t)
    logger.info("%d tickers have sufficient history", len(valid))
    return pd.Index(valid)

def write_filtered_sep(
    conn: duckdb.DuckDBPyConnection,
    tickers: pd.Index,
    path: str,
    start_date: pd.Timestamp
):
    """
    Write filtered sep_base rows for tickers since start_date to Parquet.
    """
    logger.info("Writing filtered SEP data to %s", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tickers_list = ",".join(f"'{t}'" for t in tickers)
    q = f"""
    SELECT * FROM sep_base
    WHERE ticker IN ({tickers_list}) AND date>=DATE'{start_date.date()}'
    ORDER BY ticker, date
    """
    df = conn.execute(q).fetchdf()
    df.to_parquet(path, index=False)
    logger.info("Wrote %d rows to %s", len(df), path)

def write_universe_csv(
    tickers: pd.Index,
    path: str
):
    """
    Save ticker list to CSV.
    """
    logger.info("Writing universe CSV to %s", path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({'ticker':tickers}).to_csv(path, index=False)
    logger.info("Wrote %d tickers", len(tickers))

def main():
    p = argparse.ArgumentParser(description="Filter common stock universe")
    p.add_argument('--db', default='data/kairos.duckdb')
    p.add_argument('--min-adv', type=int, default=2000000)
    p.add_argument('--min-days', type=int, default=252)
    p.add_argument('--start-date', default='1998-01-01')
    p.add_argument('--min-price', type=float, default=None)
    p.add_argument('--bucket', default='midcap_and_up')
    p.add_argument('--output-dir', default='scripts/sep_dataset/feature_sets')
    args = p.parse_args()

    start_date = validate_date(args.start_date)
    conn = duckdb.connect(args.db)

    # 1) ADV + category
    df_can = get_candidate_tickers(conn, args.min_adv)
    # 2) History length
    valid = ensure_min_trading_days(conn, df_can['ticker'], args.min_days, start_date)
    # 3) Price floor
    if args.min_price is not None:
        logger.info("Applying price floor >= %0.2f", args.min_price)
        tickers_csv = ",".join(f"'{t}'" for t in valid)
        q2 = f"""
        SELECT ticker FROM sep_base
        WHERE ticker IN ({tickers_csv})
          AND date=(SELECT MAX(date) FROM sep_base WHERE ticker=sep_base.ticker)
          AND close>={args.min_price}
        """
        df_p = conn.execute(q2).fetchdf()
        valid = valid.intersection(df_p['ticker'].tolist())
        logger.info("%d tickers after price filter", len(valid))

    # file paths
    date_str = datetime.now().strftime('%Y_%m_%d')
    sep_path = os.path.join(args.output_dir, f"{args.bucket}_features_{date_str}.parquet")
    uni_path = os.path.join(args.output_dir, f"{args.bucket}_universe_{date_str}.csv")

    # 4) Write outputs
    write_filtered_sep(conn, valid, sep_path, start_date)
    write_universe_csv(valid, uni_path)

    # 5) Create DuckDB table mid_cap_<date>
    table_name = f"mid_cap_{date_str}"
    logger.info("Creating DuckDB table %s", table_name)
    conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_parquet('{sep_path}')")
    logger.info("Table %s created in %s", table_name, args.db)

    logger.info("✅ Completed filter and table creation")

if __name__=='__main__':
    main()
