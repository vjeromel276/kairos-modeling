#!/usr/bin/env python3
"""
Merge daily SEP Parquets into sep_base DuckDB table,
then update sep_base_common either incrementally or via full rebuild.
"""

import argparse
import logging
import os
import sys
import glob
from datetime import datetime, date
import duckdb  # type: ignore

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def parse_date_from_filename(filename: str) -> date:
    base = os.path.basename(filename)
    try:
        date_part = base.replace("SHARADAR_SEP_", "").replace(".parquet", "")
        return datetime.strptime(date_part, "%Y-%m-%d").date()
    except Exception:
        raise ValueError(f"Cannot parse date from filename '{filename}'.")


def find_daily_files(daily_dir: str) -> list[str]:
    pattern = os.path.join(daily_dir, "SHARADAR_SEP_*.parquet")
    files = glob.glob(pattern)
    files_sorted = sorted(files, key=lambda p: parse_date_from_filename(p))
    logger.info(f"Found {len(files_sorted)} daily file(s) in '{daily_dir}'")
    return files_sorted


def connect_duckdb(db_path: str) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        logger.info(f"DuckDB not found at '{db_path}'. Creating new database.")
    return duckdb.connect(db_path)


def ensure_sep_base_table(conn: duckdb.DuckDBPyConnection, sample_file: str):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sep_base AS
        SELECT * FROM read_parquet(?) LIMIT 0
    """, (sample_file,))
    logger.info("Verified or created sep_base table in DuckDB.")


def get_current_max_date(conn: duckdb.DuckDBPyConnection, table: str) -> date | None:
    result = conn.execute(f"SELECT MAX(date) FROM {table}").fetchone()
    return result[0] if result and result[0] is not None else None


def merge_daily_file(conn: duckdb.DuckDBPyConnection, parquet_path: str, max_date: date | None):
    logger.info(f"Merging daily file: {parquet_path}")
    filter_clause = f"WHERE date > DATE '{max_date}'" if max_date else ""
    conn.execute(f"""
        INSERT INTO sep_base
        SELECT DISTINCT * FROM read_parquet('{parquet_path}')
        {filter_clause}
    """)
    logger.info(f"Inserted rows from {os.path.basename(parquet_path)}")
    os.remove(parquet_path)
    logger.info(f"Deleted daily file: {parquet_path}")


def update_sep_base_common_full(conn: duckdb.DuckDBPyConnection):
    logger.info("Performing full rebuild of sep_base_common...")
    conn.execute("""
        CREATE OR REPLACE TABLE sep_base_common AS
        SELECT s.*
        FROM sep_base s
        JOIN ticker_metadata_view m
          ON s.ticker = m.ticker
        WHERE m.category = 'Domestic Common Stock'
          AND m.scalemarketcap IN ('4 - Mid','5 - Large','6 - Mega')
          AND m.volumeavg1m >= 2_000_000
    """)
    logger.info("sep_base_common has been fully rebuilt.")


def update_sep_base_common_incremental(conn: duckdb.DuckDBPyConnection):
    logger.info("Performing incremental update of sep_base_common...")
    # Ensure table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sep_base_common AS
        SELECT * FROM sep_base WHERE FALSE
    """)
    max_common_date = get_current_max_date(conn, "sep_base_common")
    if max_common_date is None:
        logger.info("sep_base_common is empty; falling back to full rebuild.")
        update_sep_base_common_full(conn)
        return

    logger.info(f"Current max date in sep_base_common: {max_common_date}")
    conn.execute(f"""
        INSERT INTO sep_base_common
        SELECT s.*
        FROM sep_base s
        JOIN ticker_metadata_view m
          ON s.ticker = m.ticker
        WHERE m.category = 'Domestic Common Stock'
          AND m.scalemarketcap IN ('4 - Mid','5 - Large','6 - Mega')
          AND m.volumeavg1m >= 2_000_000
          AND s.date > DATE '{max_common_date}'
    """)
    logger.info("Inserted new rows into sep_base_common.")


def main():
    parser = argparse.ArgumentParser(
        description="Merge daily SEP Parquets into sep_base, then update sep_base_common."
    )
    parser.add_argument(
        "--update-golden",
        default="data/kairos.duckdb",
        help="Path to the DuckDB database file."
    )
    parser.add_argument(
        "--daily-dir",
        default="scripts/sep_dataset/daily_downloads/",
        help="Directory containing daily SHARADAR_SEP_*.parquet files."
    )
    parser.add_argument(
        "--rebuild-common",
        action="store_true",
        help="If set, do a full CREATE OR REPLACE rebuild of sep_base_common. "
             "Otherwise, update it incrementally."
    )
    args = parser.parse_args()

    # 1) Merge daily files
    daily_files = find_daily_files(args.daily_dir)
    if not daily_files:
        logger.warning("No daily files found. Exiting without changes.")
        sys.exit(0)

    conn = connect_duckdb(args.update_golden)
    ensure_sep_base_table(conn, daily_files[0])
    max_date = get_current_max_date(conn, "sep_base")
    if isinstance(max_date, datetime):
        max_date = max_date.date()
    logger.info(f"Current max date in sep_base: {max_date or 'None'}")

    to_merge = [
        path for path in daily_files
        if max_date is None or parse_date_from_filename(path) > max_date
    ]
    if not to_merge:
        logger.info("sep_base is already up to date; no new files to merge.")
    else:
        for path in to_merge:
            merge_daily_file(conn, path, max_date)
        logger.info("All new daily files merged successfully.")

    # 2) Update sep_base_common
    if args.rebuild_common:
        update_sep_base_common_full(conn)
    else:
        update_sep_base_common_incremental(conn)

    conn.close()


if __name__ == "__main__":
    main()
