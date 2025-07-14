#!/usr/bin/env python3
# daily_download.py
# ——————————————————————————————
# Download a single day’s SHARADAR SEP CSV via HTTP and convert it to Parquet.
# Usage: python daily_download.py --date YYYY-MM-DD

import argparse
import logging
import os
import sys
from datetime import datetime
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
API_KEY_ENV = os.getenv("NASDAQ_DATA_LINK_API_KEY")


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


def build_sep_url(date_str: str, api_key: str) -> str:
    """
    Construct the Sharadar SEP CSV URL for the given date and API key.
    """
    base = "https://data.nasdaq.com/api/v3/datatables/SHARADAR/SEP.csv"
    url = f"{base}?date={date_str}&api_key={api_key}"
    logger.info(f"Built URL: {url}")
    return url


def download_csv(url: str, csv_path: str) -> None:
    """
    Download the CSV at the given URL to csv_path, streaming in chunks.
    """
    logger.info(f"Starting CSV download to '{csv_path}'")
    try:
        with requests.get(url, stream=True) as resp:
            resp.raise_for_status()
            total_bytes = 0
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=32 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            logger.info(f"Finished CSV download ({total_bytes / (1024 * 1024):.2f} MB)")
    except Exception as e:
        logger.error(f"Error during CSV download: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        sys.exit(1)


def convert_csv_to_parquet(csv_path: str, parquet_path: str) -> None:
    """
    Read the CSV at csv_path into a DataFrame and write it out as Parquet.
    Then delete the CSV to conserve space.
    """
    logger.info(f"Reading CSV into DataFrame: '{csv_path}'")
    try:
        df = pd.read_csv(csv_path, parse_dates=["date"])
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        sys.exit(1)

    logger.info(f"CSV loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"Writing Parquet to '{parquet_path}'")
    try:
        df.to_parquet(parquet_path, index=False)
        logger.info(f"Successfully wrote Parquet: '{parquet_path}'")
    except Exception as e:
        logger.error(f"Failed to write Parquet: {e}")
        sys.exit(1)
    finally:
        # Clean up CSV
        try:
            os.remove(csv_path)
            logger.info(f"Removed temporary CSV: '{csv_path}'")
        except Exception:
            logger.warning(f"Could not delete temporary CSV: '{csv_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Download SHARADAR SEP for one date and save as Parquet."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Date to download (YYYY-MM-DD)"
    )
    args = parser.parse_args()

    # 1. Validate date format
    try:
        dt = validate_date(args.date)
    except ValueError as ve:
        logger.error(ve)
        sys.exit(1)
    date_str = dt.strftime("%Y-%m-%d")
    logger.info(f"Validated date: {date_str}")

    # 2. Ensure API key is available
    api_key =  API_KEY_ENV
    if not api_key:
        logger.error(f"Environment variable '{API_KEY_ENV}' not found.")
        sys.exit(1)

    # 3. Build URL
    url = build_sep_url(date_str, api_key)

    # 4. Define local paths
    csv_filename = f"SHARADAR_SEP_{date_str}.csv"
    parquet_filename = f"SHARADAR_SEP_{date_str}.parquet"
    csv_path = os.path.join(DAILY_DIR, csv_filename)
    parquet_path = os.path.join(DAILY_DIR, parquet_filename)

    # 5. Download CSV → Parquet
    download_csv(url, csv_path)
    convert_csv_to_parquet(csv_path, parquet_path)

    logger.info(f"Daily download complete for {date_str}: '{parquet_path}'")


if __name__ == "__main__":
    main()
