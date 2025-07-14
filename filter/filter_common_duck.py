#!/usr/bin/env python3
# filter_common_duck.py
# —-----------------------------------------
# DuckDB-native version of filter_common.py
# Pulls from: sep_base, tickers, metrics tables
# Outputs: filtered SEP Parquet + ticker universe CSV

import argparse
import os
import logging
from datetime import datetime
import duckdb
import pandas as pd
from tqdm import tqdm
from pandas.tseries.offsets import BDay

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def validate_date(date_str: str) -> datetime.date: # type: ignore
    return datetime.strptime(date_str, "%Y-%m-%d").date()

def get_candidate_tickers(conn, min_volume: int) -> pd.DataFrame:
    query = f"""
    SELECT t.ticker
    FROM tickers t
    JOIN (
        SELECT ticker, MAX(date) as latest_date
        FROM metrics
        GROUP BY ticker
    ) m_latest ON t.ticker = m_latest.ticker
    JOIN metrics m ON m.ticker = m_latest.ticker AND m.date = m_latest.latest_date
    WHERE 
        LOWER(t.category) LIKE '%common stock%'
        AND COALESCE(LOWER(t.isdelisted), 'false') IN ('false', '0', 'n', 'no')
        AND m.volumeavg1m >= {min_volume}
    """
    logger.info("Running DuckDB filter query for common stocks + volume...")
    df = conn.execute(query).fetchdf()
    logger.info(f"Tickers meeting volume + category filters: {len(df):,}")
    return df

def ensure_min_trading_days(conn, tickers: pd.Series, min_days: int, start_date: str) -> pd.Index:
    logger.info(f"Checking for >= {min_days} trading days since {start_date}...")
    valid = []
    for ticker in tqdm(tickers, desc="Evaluating history length"):
        q = f"""
        SELECT date FROM sep_base
        WHERE ticker = '{ticker}' AND date >= DATE '{start_date}'
        ORDER BY date
        """
        df = conn.execute(q).fetchdf()
        if len(df) < min_days:
            continue
        df_dates = pd.to_datetime(df["date"]).dt.normalize().drop_duplicates()
        date_range = pd.date_range(start=df_dates.min(), end=df_dates.max(), freq=BDay())
        if df_dates.isin(date_range).sum() >= min_days:
            valid.append(ticker)
    logger.info(f"Tickers with sufficient history: {len(valid):,} / {len(tickers):,}")
    return pd.Index(valid)

def write_filtered_sep(conn, tickers: pd.Index, path: str, start_date: str):
    logger.info("Extracting filtered SEP data from DuckDB...")
    ticker_list = ",".join([f"'{t}'" for t in tickers])
    query = f"""
        SELECT * FROM sep_base
        WHERE ticker IN ({ticker_list})
        AND date >= DATE '{start_date}'
        ORDER BY ticker, date
    """
    df = conn.execute(query).fetchdf()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info(f"Wrote filtered SEP Parquet: {path} ({len(df):,} rows)")

def write_universe_csv(tickers: pd.Index, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(path, index=False)
    logger.info(f"Wrote ticker universe CSV: {path} ({len(tickers):,} tickers)")

def main():
    p = argparse.ArgumentParser(description="Filter common stocks using DuckDB-native pipeline.")
    p.add_argument('--db', default="data/karios.duckdb", help="Path to DuckDB database")
    p.add_argument('--min-volume', type=int, default=1000000)
    p.add_argument('--min-days', type=int, default=252)
    p.add_argument('--start-date', default="1998-01-01")
    p.add_argument('--bucket', default="midcap_and_up")
    p.add_argument('--output-dir', default="feature_sets")
    args = p.parse_args()

    start_date = validate_date(args.start_date)
    conn = duckdb.connect(args.db)

    df_candidates = get_candidate_tickers(conn, args.min_volume)
    valid = ensure_min_trading_days(conn, df_candidates["ticker"], args.min_days, args.start_date)

    date_str = datetime.now().strftime('%Y-%m-%d')
    sep_path = os.path.join(args.output_dir, f"{args.bucket}_tickers_features_{date_str}.parquet")
    uni_path = os.path.join(args.output_dir, f"{args.bucket}_ticker_universe_{date_str}.csv")

    write_filtered_sep(conn, valid, sep_path, args.start_date)
    write_universe_csv(valid, uni_path)
    logger.info("✅ filter_common_duck.py completed successfully.")

if __name__ == "__main__":
    main()
