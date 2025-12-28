#!/usr/bin/env python3
# daily_download.py
# ——————————————————————————————
# Download a single day's SHARADAR SEP and DAILY data via HTTP and convert to Parquet.
#
# Tables downloaded:
#   - SEP: Daily stock prices
#   - DAILY: Daily fundamental ratios (PE, PB, PS, EV/EBITDA, market cap, etc.)
#
# Usage: 
#   python daily_download.py --date YYYY-MM-DD
#   python daily_download.py --date YYYY-MM-DD --tables SEP DAILY

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict
import requests
import pandas as pd

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Directory where daily Parquets (and temporary CSVs) will live
DAILY_DIR = os.path.join("scripts/sep_dataset", "daily_downloads")
os.makedirs(DAILY_DIR, exist_ok=True)

# API key environment variable name
API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"

# Table configurations
TABLES: Dict[str, Dict] = {
    "SEP": {
        "description": "Daily stock prices",
        "date_field": "date",
        "date_columns": ["date"],
    },
    "DAILY": {
        "description": "Daily fundamental ratios (PE, PB, PS, EV/EBITDA)",
        "date_field": "date",
        "date_columns": ["date", "lastupdated"],
    },
}

# Default tables to download
DEFAULT_TABLES = ["SEP", "DAILY"]


def validate_date(date_str: str) -> datetime:
    """
    Validate that the supplied date string is in YYYY-MM-DD format.
    Returns a datetime.date on success; raises ValueError otherwise.
    """
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d").date()
        return dt
    except ValueError:
        raise ValueError(f"Invalid date format: '{date_str}'. Use YYYY-MM-DD.")


def build_url(table: str, date_str: str, api_key: str) -> str:
    """
    Construct the Sharadar table CSV URL for the given date and API key.
    """
    base = f"https://data.nasdaq.com/api/v3/datatables/SHARADAR/{table}.csv"
    date_field = TABLES[table]["date_field"]
    url = f"{base}?{date_field}={date_str}&api_key={api_key}"
    # Log with masked API key
    masked_key = api_key[:4] + "****" + api_key[-4:] if len(api_key) > 8 else "****"
    logger.info(f"Built URL: {base}?{date_field}={date_str}&api_key={masked_key}")
    return url


def download_csv(url: str, csv_path: str) -> bool:
    """
    Download the CSV at the given URL to csv_path, streaming in chunks.
    Returns True if successful and data was found, False if no data.
    """
    logger.info(f"Starting CSV download to '{csv_path}'")
    try:
        with requests.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            total_bytes = 0
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            
            if total_bytes < 100:
                logger.warning(f"Response very small ({total_bytes} bytes), may be empty or header-only")
            
            logger.info(f"Finished CSV download ({total_bytes / (1024 * 1024):.2f} MB)")
            return True
            
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logger.warning(f"No data found (404) - may be holiday or weekend")
            return False
        logger.error(f"HTTP error during CSV download: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return False
    except Exception as e:
        logger.error(f"Error during CSV download: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return False


def convert_csv_to_parquet(csv_path: str, parquet_path: str, date_columns: List[str]) -> int:
    """
    Read the CSV at csv_path into a DataFrame and write it out as Parquet.
    Then delete the CSV to conserve space.
    Returns row count, or 0 if empty/failed.
    """
    logger.info(f"Reading CSV into DataFrame: '{csv_path}'")
    try:
        df = pd.read_csv(csv_path, parse_dates=date_columns, low_memory=False)
    except pd.errors.EmptyDataError:
        logger.warning("CSV is empty (no data for this date)")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return 0
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return 0

    if df.empty:
        logger.warning("CSV loaded but DataFrame is empty")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return 0

    logger.info(f"CSV loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Writing Parquet to '{parquet_path}'")
    
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Successfully wrote Parquet: '{parquet_path}'")
    except Exception as e:
        logger.error(f"Failed to write Parquet: {e}")
        return 0
    finally:
        # Clean up CSV
        try:
            os.remove(csv_path)
            logger.info(f"Removed temporary CSV: '{csv_path}'")
        except Exception:
            logger.warning(f"Could not delete temporary CSV: '{csv_path}'")
    
    return len(df)


def download_table(table: str, date_str: str, api_key: str) -> int:
    """
    Download a single table for a single date.
    Returns row count (0 if no data or error).
    """
    config = TABLES[table]
    logger.info(f"\n--- Downloading {table}: {config['description']} ---")
    
    # Define local paths
    csv_filename = f"SHARADAR_{table}_{date_str}.csv"
    parquet_filename = f"SHARADAR_{table}_{date_str}.parquet"
    csv_path = os.path.join(DAILY_DIR, csv_filename)
    parquet_path = os.path.join(DAILY_DIR, parquet_filename)
    
    # Skip if already exists
    if os.path.exists(parquet_path):
        logger.info(f"Already exists, skipping: '{parquet_path}'")
        return -1  # Already exists
    
    # Build URL and download
    url = build_url(table, date_str, api_key)
    
    if not download_csv(url, csv_path):
        return 0
    
    # Convert to Parquet
    rows = convert_csv_to_parquet(csv_path, parquet_path, config["date_columns"])
    
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Download SHARADAR SEP and DAILY for one date and save as Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download both SEP and DAILY (default)
  python daily_download.py --date 2025-12-26
  
  # Download only SEP
  python daily_download.py --date 2025-12-26 --tables SEP
  
  # Download only DAILY (to fill gaps)
  python daily_download.py --date 2025-12-26 --tables DAILY
"""
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to download (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLES.keys()),
        default=DEFAULT_TABLES,
        help=f"Tables to download (default: {' '.join(DEFAULT_TABLES)})"
    )
    args = parser.parse_args()

    # 1. Validate date format
    try:
        dt = validate_date(args.date)
    except ValueError as ve:
        logger.error(ve)
        sys.exit(1)
    date_str = dt.strftime("%Y-%m-%d")
    
    # 2. Ensure API key is available
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        logger.error(f"Environment variable '{API_KEY_ENV}' not found.")
        sys.exit(1)

    # 3. Download each table
    logger.info(f"\n{'='*60}")
    logger.info(f"SHARADAR DAILY DOWNLOAD")
    logger.info(f"{'='*60}")
    logger.info(f"Date: {date_str}")
    logger.info(f"Tables: {', '.join(args.tables)}")
    logger.info(f"{'='*60}")
    
    results = {}
    for table in args.tables:
        rows = download_table(table, date_str, api_key)
        results[table] = rows
    
    # 4. Summary
    logger.info(f"\n{'='*60}")
    logger.info("DOWNLOAD SUMMARY")
    logger.info(f"{'='*60}")
    
    success = True
    for table, rows in results.items():
        if rows == -1:
            status = "SKIPPED (already exists)"
        elif rows == 0:
            status = "NO DATA (holiday/weekend?)"
        else:
            status = f"{rows:,} rows"
        logger.info(f"  {table:8}: {status}")
        
        if rows == 0:
            success = False
    
    logger.info(f"{'='*60}")
    logger.info(f"Daily download complete for {date_str}")
    
    # Return 0 even if no data (could be weekend/holiday)
    sys.exit(0)


if __name__ == "__main__":
    main()