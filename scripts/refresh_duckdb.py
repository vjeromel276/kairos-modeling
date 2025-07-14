#!/usr/bin/env python3
# refresh_duckdb.py
# ------------------------------------------------------------
# Incrementally update DuckDB metrics table (--pull-metrics)
# and/or reload CSVs like sf1.csv, sf2.csv from disk (--reload)

import argparse
import logging
import os
import sys
from datetime import datetime
import duckdb
import pandas as pd
import requests
from io import StringIO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
NDL_METRICS_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/METRICS.csv"


def pull_new_metrics(conn: duckdb.DuckDBPyConnection):
    if not API_KEY:
        logger.error("Missing NASDAQ_DATA_LINK_API_KEY in environment.")
        sys.exit(1)

    latest_date = conn.execute("SELECT MAX(date) FROM metrics").fetchone()[0] # type: ignore
    if latest_date is None:
        logger.error("metrics table exists but has no rows.")
        sys.exit(1)

    logger.info(f"Latest metrics date in DuckDB: {latest_date}")
    logger.info("Requesting new data from NDL API...")

    params = {
        "api_key": API_KEY,
        "date.gt": latest_date.strftime("%Y-%m-%d")
    }

    try:
        response = requests.get(NDL_METRICS_URL, params=params)
        response.raise_for_status()
        new_data = pd.read_csv(StringIO(response.text))
    except Exception as e:
        logger.error(f"Failed to pull from API: {e}")
        sys.exit(1)

    if new_data.empty:
        logger.info("No new metrics rows found.")
        return

    logger.info(f"Pulled {new_data.shape[0]:,} new metrics rows.")
    conn.register("__new_metrics__", new_data)
    conn.execute("INSERT INTO metrics SELECT * FROM __new_metrics__")
    logger.info("Appended new metrics to DuckDB.")


def reload_csv_tables(conn: duckdb.DuckDBPyConnection, csv_dir: str):
    files = [f for f in os.listdir(csv_dir) if f.lower().endswith(".csv")]
    if not files:
        logger.info(f"No CSVs found in '{csv_dir}'. Skipping reload.")
        return

    for filename in files:
        table_name = os.path.splitext(filename)[0].lower()
        full_path = os.path.join(csv_dir, filename)
        logger.info(f"Reloading '{table_name}' from {full_path}...")

        try:
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{full_path}', HEADER=TRUE)")
            row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0] # type: ignore
            logger.info(f"Created table '{table_name}' with {row_count:,} rows")
            os.remove(full_path)
            logger.info(f"Deleted CSV: {full_path}")
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Refresh DuckDB from NDL or CSVs")
    parser.add_argument("--db", default="data/karios.duckdb", help="Path to DuckDB file")
    parser.add_argument("--pull-metrics", action="store_true", help="Fetch new metrics from NDL API")
    parser.add_argument("--reload", action="store_true", help="Reload any *.csv files found in original_data/")
    parser.add_argument("--csv-dir", default="original_data", help="Directory to look for reloadable CSVs")
    args = parser.parse_args()

    conn = duckdb.connect(args.db)
    logger.info(f"Connected to {args.db}")

    if args.pull_metrics:
        pull_new_metrics(conn)

    if args.reload:
        reload_csv_tables(conn, args.csv_dir)

    if not args.pull_metrics and not args.reload:
        logger.warning("No action taken. Use --pull-metrics or --reload.")


if __name__ == "__main__":
    main()
