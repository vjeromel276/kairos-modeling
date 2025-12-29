#!/usr/bin/env python3
"""
merge_daily_download_duck.py
============================
Merge daily SEP and DAILY Parquets into DuckDB tables,
then update derived tables (sep_base_common).

Tables handled:
  - SEP → sep_base (incremental price data)
  - DAILY → daily (incremental fundamental ratios: PE, PB, PS, EV/EBITDA, etc.)

Usage:
  python scripts/merge_daily_download_duck.py --update-golden data/kairos.duckdb --daily-dir scripts/sep_dataset/daily_downloads/
"""

import argparse
import logging
import os
import sys
import glob
from datetime import datetime, date
from typing import Optional, List, Dict
import duckdb

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Table configurations
# Maps file prefix to DuckDB table name and date column
TABLE_CONFIG: Dict[str, Dict] = {
    "SHARADAR_SEP": {
        "table_name": "sep_base",
        "date_column": "date",
        "description": "Daily stock prices",
    },
    "SHARADAR_DAILY": {
        "table_name": "daily",
        "date_column": "date",
        "description": "Daily fundamental ratios",
    },
}


def parse_date_from_filename(filename: str) -> date:
    """Extract date from filename like SHARADAR_SEP_2025-12-26.parquet"""
    base = os.path.basename(filename)
    try:
        # Handle both SEP and DAILY formats
        # SHARADAR_SEP_2025-12-26.parquet → 2025-12-26
        # SHARADAR_DAILY_2025-12-26.parquet → 2025-12-26
        parts = base.replace(".parquet", "").split("_")
        date_part = parts[-1]  # Last part is the date
        return datetime.strptime(date_part, "%Y-%m-%d").date()
    except Exception:
        raise ValueError(f"Cannot parse date from filename '{filename}'.")


def get_file_prefix(filename: str) -> str:
    """Extract table prefix from filename like SHARADAR_SEP_2025-12-26.parquet"""
    base = os.path.basename(filename)
    # SHARADAR_SEP_2025-12-26.parquet → SHARADAR_SEP
    parts = base.replace(".parquet", "").split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    raise ValueError(f"Cannot parse prefix from filename '{filename}'.")


def find_daily_files(daily_dir: str) -> Dict[str, List[str]]:
    """
    Find all parquet files in the directory, grouped by table type.
    Returns: {"SHARADAR_SEP": [file1, file2, ...], "SHARADAR_DAILY": [...]}
    """
    all_files = glob.glob(os.path.join(daily_dir, "SHARADAR_*.parquet"))
    
    grouped: Dict[str, List[str]] = {prefix: [] for prefix in TABLE_CONFIG.keys()}
    
    for f in all_files:
        try:
            prefix = get_file_prefix(f)
            if prefix in grouped:
                grouped[prefix].append(f)
        except ValueError as e:
            logger.warning(f"Skipping file: {e}")
    
    # Sort each group by date
    for prefix in grouped:
        grouped[prefix] = sorted(grouped[prefix], key=lambda p: parse_date_from_filename(p))
        if grouped[prefix]:
            logger.info(f"Found {len(grouped[prefix])} {prefix} file(s)")
    
    return grouped


def connect_duckdb(db_path: str) -> duckdb.DuckDBPyConnection:
    if not os.path.exists(db_path):
        logger.info(f"DuckDB not found at '{db_path}'. Creating new database.")
    return duckdb.connect(db_path)


def ensure_table_exists(conn: duckdb.DuckDBPyConnection, table_name: str, sample_file: str):
    """Create table if it doesn't exist, using sample file for schema."""
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    
    if table_name not in tables:
        logger.info(f"Creating table '{table_name}' from sample file schema...")
        conn.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_parquet('{sample_file}') LIMIT 0
        """)
        logger.info(f"Created empty table '{table_name}'")
    else:
        logger.info(f"Table '{table_name}' already exists")


def get_current_max_date(conn: duckdb.DuckDBPyConnection, table: str, date_column: str) -> Optional[date]:
    """Get the maximum date currently in the table."""
    try:
        result = conn.execute(f"SELECT MAX({date_column}) FROM {table}").fetchone()
        if result and result[0] is not None:
            max_val = result[0]
            if isinstance(max_val, datetime):
                return max_val.date()
            return max_val
    except Exception as e:
        logger.warning(f"Could not get max date from {table}: {e}")
    return None


def merge_parquet_file(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    table_name: str,
    date_column: str,
    max_date: Optional[date]
) -> int:
    """
    Merge a single parquet file into the target table.
    Returns number of rows inserted.
    """
    logger.info(f"Merging: {os.path.basename(parquet_path)} → {table_name}")
    
    # Count rows before
    try:
        before_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    except Exception:
        before_count = 0
    
    # Build filter clause
    if max_date:
        filter_clause = f"WHERE {date_column} > DATE '{max_date}'"
    else:
        filter_clause = ""
    
    # Insert new rows
    conn.execute(f"""
        INSERT INTO {table_name}
        SELECT DISTINCT * FROM read_parquet('{parquet_path}')
        {filter_clause}
    """)
    
    # Count rows after
    after_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    inserted = after_count - before_count
    
    logger.info(f"  Inserted {inserted:,} rows from {os.path.basename(parquet_path)}")
    
    # Delete the processed file
    os.remove(parquet_path)
    logger.info(f"  Deleted: {parquet_path}")
    
    return inserted


def merge_table_files(
    conn: duckdb.DuckDBPyConnection,
    files: List[str],
    config: Dict
) -> int:
    """Merge all files for a single table type. Returns total rows inserted."""
    if not files:
        return 0
    
    table_name = config["table_name"]
    date_column = config["date_column"]
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Merging {len(files)} files into '{table_name}'")
    logger.info(f"{'='*50}")
    
    # Ensure table exists
    ensure_table_exists(conn, table_name, files[0])
    
    # Get current max date
    max_date = get_current_max_date(conn, table_name, date_column)
    if isinstance(max_date, datetime):
        max_date = max_date.date()
    logger.info(f"Current max date in {table_name}: {max_date or 'None'}")
    
    # Filter to only files with dates > max_date
    files_to_merge = []
    for f in files:
        file_date = parse_date_from_filename(f)
        if max_date is None or file_date > max_date:
            files_to_merge.append(f)
        else:
            logger.info(f"  Skipping {os.path.basename(f)} (date {file_date} <= {max_date})")
            # Still delete old files
            os.remove(f)
    
    if not files_to_merge:
        logger.info(f"No new files to merge for {table_name}")
        return 0
    
    # Merge each file
    total_inserted = 0
    for f in files_to_merge:
        inserted = merge_parquet_file(conn, f, table_name, date_column, max_date)
        total_inserted += inserted
    
    # Report final state
    new_max = get_current_max_date(conn, table_name, date_column)
    total_rows = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    logger.info(f"✓ {table_name}: {total_rows:,} total rows, max date = {new_max}")
    
    return total_inserted


def update_sep_base_common_full(conn: duckdb.DuckDBPyConnection):
    """Full rebuild of sep_base_common from sep_base."""
    logger.info("\nPerforming full rebuild of sep_base_common...")
    
    # Check if required view exists
    try:
        conn.execute("SELECT 1 FROM ticker_metadata_view LIMIT 1")
    except Exception:
        logger.warning("ticker_metadata_view not found, skipping sep_base_common update")
        return
    
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
    
    count = conn.execute("SELECT COUNT(*) FROM sep_base_common").fetchone()[0]
    logger.info(f"✓ sep_base_common rebuilt: {count:,} rows")


def update_sep_base_common_incremental(conn: duckdb.DuckDBPyConnection):
    """Incremental update of sep_base_common."""
    logger.info("\nPerforming incremental update of sep_base_common...")
    
    # Check if table exists
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    
    if "sep_base_common" not in tables:
        logger.info("sep_base_common doesn't exist; doing full rebuild")
        update_sep_base_common_full(conn)
        return
    
    # Check if required view exists
    try:
        conn.execute("SELECT 1 FROM ticker_metadata_view LIMIT 1")
    except Exception:
        logger.warning("ticker_metadata_view not found, skipping sep_base_common update")
        return
    
    max_common_date = get_current_max_date(conn, "sep_base_common", "date")
    if max_common_date is None:
        logger.info("sep_base_common is empty; falling back to full rebuild")
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
    
    new_max = get_current_max_date(conn, "sep_base_common", "date")
    logger.info(f"✓ sep_base_common updated, new max date: {new_max}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge daily SEP and DAILY Parquets into DuckDB tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Merge all files in default directory
  python merge_daily_download_duck.py --update-golden data/kairos.duckdb
  
  # Merge with full rebuild of sep_base_common
  python merge_daily_download_duck.py --update-golden data/kairos.duckdb --rebuild-common
"""
    )
    parser.add_argument(
        "--update-golden",
        default="data/kairos.duckdb",
        help="Path to the DuckDB database file."
    )
    parser.add_argument(
        "--daily-dir",
        default="scripts/sep_dataset/daily_downloads/",
        help="Directory containing daily SHARADAR_*.parquet files."
    )
    parser.add_argument(
        "--rebuild-common",
        action="store_true",
        help="Full rebuild of sep_base_common instead of incremental update."
    )
    args = parser.parse_args()

    logger.info(f"\n{'='*60}")
    logger.info("MERGE DAILY DOWNLOADS")
    logger.info(f"{'='*60}")
    logger.info(f"Database: {args.update_golden}")
    logger.info(f"Directory: {args.daily_dir}")
    
    # Find all files grouped by table type
    grouped_files = find_daily_files(args.daily_dir)
    
    total_files = sum(len(files) for files in grouped_files.values())
    if total_files == 0:
        logger.warning("No daily files found. Exiting without changes.")
        sys.exit(0)
    
    # Connect to database
    conn = connect_duckdb(args.update_golden)
    
    # Merge each table type
    total_inserted = 0
    for prefix, config in TABLE_CONFIG.items():
        files = grouped_files.get(prefix, [])
        if files:
            inserted = merge_table_files(conn, files, config)
            total_inserted += inserted
    
    # Update sep_base_common
    if "sep_base" in [c["table_name"] for c in TABLE_CONFIG.values()]:
        if args.rebuild_common:
            update_sep_base_common_full(conn)
        else:
            update_sep_base_common_incremental(conn)
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("MERGE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total rows inserted: {total_inserted:,}")
    
    # Show current state of tables
    for prefix, config in TABLE_CONFIG.items():
        table_name = config["table_name"]
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            max_date = get_current_max_date(conn, table_name, config["date_column"])
            logger.info(f"  {table_name}: {count:,} rows, max date = {max_date}")
        except Exception:
            logger.info(f"  {table_name}: (not found)")
    
    conn.close()


if __name__ == "__main__":
    main()