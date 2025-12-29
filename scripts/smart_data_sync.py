#!/usr/bin/env python3
"""
smart_data_sync.py
==================
Intelligently sync Sharadar data tables with the API.

For each table:
1. Check current max date in local DuckDB
2. Query API to see if newer data exists
3. Download only if new data is available
4. Merge into DuckDB

Tables supported:
- SEP: Daily stock prices (date field: date)
- DAILY: Daily fundamental ratios (date field: date)
- SF1: Quarterly fundamentals (date field: lastupdated)
- SF2: Insider transactions (date field: filingdate)

Usage:
    # Check and sync all tables
    python smart_data_sync.py --db data/kairos.duckdb
    
    # Check only (don't download)
    python smart_data_sync.py --db data/kairos.duckdb --check-only
    
    # Sync specific tables
    python smart_data_sync.py --db data/kairos.duckdb --tables SEP DAILY
    
    # Force download even if up to date
    python smart_data_sync.py --db data/kairos.duckdb --force

Environment:
    NASDAQ_DATA_LINK_API_KEY: Your Nasdaq Data Link API key
"""

import argparse
import logging
import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple
import requests
import pandas as pd
import duckdb

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# API configuration
API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"

# Table configurations
TABLES = {
    "SEP": {
        "db_table": "sep_base",
        "date_field": "date",           # Field to filter API by
        "db_date_field": "date",        # Field in local DB to check max
        "description": "Daily stock prices",
        "frequency": "daily",
    },
    "DAILY": {
        "db_table": "daily",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Daily fundamental ratios (PE, PB, PS, EV/EBITDA)",
        "frequency": "daily",
    },
    "SF1": {
        "db_table": "sf1",
        "date_field": "lastupdated",    # Filter API by lastupdated.gte
        "db_date_field": "lastupdated", # Check max lastupdated locally
        "description": "Quarterly/Annual fundamentals",
        "frequency": "quarterly",
        "use_gte": True,
    },
    "SF2": {
        "db_table": "sf2",
        "date_field": "filingdate",
        "db_date_field": "filingdate",
        "description": "Insider transactions",
        "frequency": "daily",
        "use_gte": True,
    },
}

# Temp directory for downloads
TEMP_DIR = "scripts/sep_dataset/daily_downloads"


def get_api_key() -> str:
    """Get API key from environment."""
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        logger.error(f"Environment variable '{API_KEY_ENV}' not set.")
        sys.exit(1)
    return api_key


def get_local_max_date(conn: duckdb.DuckDBPyConnection, table_config: Dict) -> Optional[date]:
    """Get the maximum date for a table in local DuckDB."""
    db_table = table_config["db_table"]
    date_field = table_config["db_date_field"]
    
    # Check if table exists
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    if db_table not in tables:
        logger.info(f"  Table '{db_table}' does not exist locally")
        return None
    
    try:
        result = conn.execute(f"SELECT MAX({date_field}) FROM {db_table}").fetchone()
        if result and result[0]:
            max_val = result[0]
            if isinstance(max_val, datetime):
                return max_val.date()
            elif isinstance(max_val, date):
                return max_val
            else:
                return datetime.strptime(str(max_val)[:10], "%Y-%m-%d").date()
        return None
    except Exception as e:
        logger.warning(f"  Error getting max date from {db_table}: {e}")
        return None


def check_api_for_new_data(
    table_name: str, 
    table_config: Dict, 
    local_max: Optional[date],
    api_key: str
) -> Tuple[bool, Optional[date], int]:
    """
    Check if API has data newer than local_max.
    
    Returns: (has_new_data, api_max_date, estimated_rows)
    """
    date_field = table_config["date_field"]
    
    # Build URL to check for latest data
    # Query for data newer than local_max
    url = f"{BASE_URL}/{table_name}.json?"
    
    if local_max:
        # Check if anything newer than local max exists
        check_date = local_max + timedelta(days=1)
        url += f"{date_field}.gte={check_date.strftime('%Y-%m-%d')}&"
    
    url += f"qopts.columns={date_field}&"
    url += f"qopts.per_page=100&"
    url += f"api_key={api_key}"
    
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        
        # Check for data
        rows = data.get("datatable", {}).get("data", [])
        
        if not rows:
            # No new data found - we're up to date
            return False, local_max, 0  # Return local_max as api_max to show "up to date"
        
        # Find max date from all returned rows
        api_max = None
        for row in rows:
            if row[0]:
                row_date = datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
                if api_max is None or row_date > api_max:
                    api_max = row_date
        
        if api_max is None:
            return False, local_max, 0
        
        # New data exists
        if local_max is None:
            return True, api_max, -1  # -1 means unknown count, need full download
        
        return True, api_max, -1
        
    except Exception as e:
        logger.warning(f"  Error checking API for {table_name}: {e}")
        return False, None, 0


def count_new_records(
    table_name: str,
    table_config: Dict,
    since_date: date,
    api_key: str
) -> int:
    """Count how many records are available since a given date."""
    date_field = table_config["date_field"]
    use_gte = table_config.get("use_gte", False)
    
    url = f"{BASE_URL}/{table_name}.json?"
    
    if use_gte:
        url += f"{date_field}.gte={since_date.strftime('%Y-%m-%d')}&"
    else:
        url += f"{date_field}.gt={since_date.strftime('%Y-%m-%d')}&"
    
    url += f"qopts.columns={date_field}&"  # Just get one column to minimize data
    url += f"api_key={api_key}"
    
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        
        # Get total count from metadata
        meta = data.get("meta", {})
        total = meta.get("total_count", len(data.get("datatable", {}).get("data", [])))
        return total
        
    except Exception as e:
        logger.warning(f"  Error counting records for {table_name}: {e}")
        return -1


def download_new_data(
    table_name: str,
    table_config: Dict,
    since_date: Optional[date],
    api_key: str,
    output_dir: str
) -> Optional[str]:
    """
    Download new data from API to a parquet file.
    Returns path to downloaded file, or None if failed.
    """
    date_field = table_config["date_field"]
    use_gte = table_config.get("use_gte", False)
    
    # Build download URL
    url = f"{BASE_URL}/{table_name}.csv?"
    
    if since_date:
        if use_gte:
            # Include the boundary date for gte
            url += f"{date_field}.gte={since_date.strftime('%Y-%m-%d')}&"
        else:
            # Exclude the boundary date for gt
            next_date = since_date + timedelta(days=1)
            url += f"{date_field}.gte={next_date.strftime('%Y-%m-%d')}&"
    
    url += f"api_key={api_key}"
    
    # Output paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"SHARADAR_{table_name}_sync_{timestamp}.csv")
    parquet_path = os.path.join(output_dir, f"SHARADAR_{table_name}_sync_{timestamp}.parquet")
    
    logger.info(f"  Downloading {table_name} data...")
    
    try:
        # Download CSV
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            total_bytes = 0
            with open(csv_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        f.write(chunk)
                        total_bytes += len(chunk)
            
            logger.info(f"  Downloaded {total_bytes / (1024*1024):.1f} MB")
        
        # Convert to parquet
        logger.info(f"  Converting to parquet...")
        
        # Determine date columns based on table
        date_columns = {
            "SEP": ["date"],
            "DAILY": ["date", "lastupdated"],
            "SF1": ["datekey", "reportperiod", "lastupdated"],
            "SF2": ["filingdate", "transactiondate"],
        }.get(table_name, [])
        
        df = pd.read_csv(csv_path, parse_dates=date_columns, low_memory=False)
        
        if df.empty:
            logger.warning(f"  No data in downloaded file")
            os.remove(csv_path)
            return None
        
        logger.info(f"  Downloaded {len(df):,} rows")
        
        df.to_parquet(parquet_path, index=False)
        os.remove(csv_path)
        
        return parquet_path
        
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        return None


def merge_parquet_to_db(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    table_config: Dict
) -> int:
    """Merge downloaded parquet into DuckDB table. Returns rows inserted."""
    db_table = table_config["db_table"]
    date_field = table_config["db_date_field"]
    
    # Check if table exists
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    
    if db_table not in tables:
        # Create table from parquet
        logger.info(f"  Creating table '{db_table}'...")
        conn.execute(f"""
            CREATE TABLE {db_table} AS
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        count = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
        os.remove(parquet_path)
        return count
    
    # Get current max date
    local_max = get_local_max_date(conn, table_config)
    
    # Insert only new rows
    before_count = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
    
    if local_max:
        filter_clause = f"WHERE {date_field} > '{local_max}'"
    else:
        filter_clause = ""
    
    conn.execute(f"""
        INSERT INTO {db_table}
        SELECT DISTINCT * FROM read_parquet('{parquet_path}')
        {filter_clause}
    """)
    
    after_count = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
    inserted = after_count - before_count
    
    os.remove(parquet_path)
    return inserted


def sync_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    table_config: Dict,
    api_key: str,
    temp_dir: str,
    check_only: bool = False,
    force: bool = False
) -> Dict:
    """
    Sync a single table. Returns status dict.
    """
    result = {
        "table": table_name,
        "db_table": table_config["db_table"],
        "local_max": None,
        "api_max": None,
        "has_new_data": False,
        "rows_added": 0,
        "status": "unknown",
    }
    
    logger.info(f"\n{'='*50}")
    logger.info(f"{table_name}: {table_config['description']}")
    logger.info(f"{'='*50}")
    
    # Get local max date
    local_max = get_local_max_date(conn, table_config)
    result["local_max"] = local_max
    logger.info(f"  Local max date: {local_max or 'No data'}")
    
    # Check API for new data
    has_new, api_max, est_rows = check_api_for_new_data(table_name, table_config, local_max, api_key)
    result["api_max"] = api_max
    result["has_new_data"] = has_new
    
    if api_max:
        logger.info(f"  API max date: {api_max}")
    
    if not has_new and not force:
        if local_max and api_max and local_max >= api_max:
            logger.info(f"  ✓ Already up to date")
            result["status"] = "up_to_date"
        elif not api_max:
            logger.info(f"  ⚠ Could not determine API status")
            result["status"] = "check_failed"
        else:
            logger.info(f"  ✓ No new data available")
            result["status"] = "up_to_date"
        return result
    
    if force:
        logger.info(f"  Force download requested")
    else:
        logger.info(f"  New data available!")
    
    if check_only:
        logger.info(f"  (check-only mode, skipping download)")
        result["status"] = "needs_update"
        return result
    
    # Download new data
    parquet_path = download_new_data(
        table_name, table_config, local_max, api_key, temp_dir
    )
    
    if not parquet_path:
        result["status"] = "download_failed"
        return result
    
    # Merge to database
    rows_added = merge_parquet_to_db(conn, parquet_path, table_config)
    result["rows_added"] = rows_added
    
    # Get new max date
    new_max = get_local_max_date(conn, table_config)
    result["local_max"] = new_max
    
    logger.info(f"  ✓ Added {rows_added:,} rows, new max date: {new_max}")
    result["status"] = "updated"
    
    return result


def print_summary(results: List[Dict]):
    """Print summary of sync results."""
    logger.info(f"\n{'='*60}")
    logger.info("SYNC SUMMARY")
    logger.info(f"{'='*60}")
    
    for r in results:
        status_icon = {
            "up_to_date": "✓",
            "updated": "✓",
            "needs_update": "⚠",
            "download_failed": "✗",
            "check_failed": "?",
        }.get(r["status"], "?")
        
        # Format local_max properly
        local_str = str(r['local_max']) if r['local_max'] else 'None'
        
        logger.info(f"{status_icon} {r['table']:8} ({r['db_table']:12}) | "
                   f"Local: {local_str:12} | "
                   f"Status: {r['status']}")
        
        if r["rows_added"] > 0:
            logger.info(f"  └─ Added {r['rows_added']:,} rows")
    
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Sharadar data sync",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all tables for new data and sync
  python smart_data_sync.py --db data/kairos.duckdb
  
  # Check only (show what would be updated)
  python smart_data_sync.py --db data/kairos.duckdb --check-only
  
  # Sync only SEP and DAILY
  python smart_data_sync.py --db data/kairos.duckdb --tables SEP DAILY
  
  # Force re-download even if up to date
  python smart_data_sync.py --db data/kairos.duckdb --tables SF1 --force
"""
    )
    
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLES.keys()),
        default=list(TABLES.keys()),
        help=f"Tables to sync (default: all)"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check for new data, don't download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if local data appears up to date"
    )
    parser.add_argument(
        "--temp-dir",
        default=TEMP_DIR,
        help=f"Temp directory for downloads (default: {TEMP_DIR})"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = get_api_key()
    
    # Ensure temp directory exists
    os.makedirs(args.temp_dir, exist_ok=True)
    
    # Connect to database
    logger.info(f"\n{'='*60}")
    logger.info("SHARADAR DATA SYNC")
    logger.info(f"{'='*60}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Tables: {', '.join(args.tables)}")
    logger.info(f"Mode: {'Check only' if args.check_only else 'Sync'}")
    
    conn = duckdb.connect(args.db)
    
    # Sync each table
    results = []
    for table_name in args.tables:
        table_config = TABLES[table_name]
        result = sync_table(
            conn=conn,
            table_name=table_name,
            table_config=table_config,
            api_key=api_key,
            temp_dir=args.temp_dir,
            check_only=args.check_only,
            force=args.force
        )
        results.append(result)
    
    conn.close()
    
    # Print summary
    print_summary(results)
    
    # Exit with error if any syncs failed
    failed = any(r["status"] in ("download_failed", "check_failed") for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())