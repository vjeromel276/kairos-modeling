#!/usr/bin/env python3
"""
scripts/smart_data_sync.py
==================
Intelligently sync Sharadar data tables with the API.

FIXED: Now handles pagination (API returns max 10,000 rows per request)

For each table:
1. Check current max date in local DuckDB
2. Query API to see if newer data exists
3. Download only if new data is available (with pagination)
4. Merge into DuckDB

Tables supported:
- SEP: Daily stock prices (date field: date)
- DAILY: Daily fundamental ratios (date field: date)
- SF1: Quarterly fundamentals (date field: lastupdated)
- SF2: Insider transactions (date field: filingdate)

Usage:
    # Check and sync all tables
    python scripts/smart_data_sync.py --db data/kairos.duckdb
    
    # Check only (don't download)
    python scripts/smart_data_sync.py --db data/kairos.duckdb --check-only
    
    # Sync specific tables
    python scripts/smart_data_sync.py --db data/kairos.duckdb --tables SEP DAILY
    
    # Force download even if up to date
    python scripts/smart_data_sync.py --db data/kairos.duckdb --force

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
import io

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
        "date_columns": ["date"],
    },
    "DAILY": {
        "db_table": "daily",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Daily fundamental ratios (PE, PB, PS, EV/EBITDA)",
        "frequency": "daily",
        "date_columns": ["date", "lastupdated"],
    },
    "SF1": {
        "db_table": "sf1",
        "date_field": "lastupdated",    # Filter API by lastupdated.gte
        "db_date_field": "lastupdated", # Check max lastupdated locally
        "description": "Quarterly/Annual fundamentals",
        "frequency": "quarterly",
        "use_gte": True,
        "date_columns": ["datekey", "reportperiod", "lastupdated"],
    },
    "SF2": {
        "db_table": "sf2",
        "date_field": "filingdate",
        "db_date_field": "filingdate",
        "description": "Insider transactions",
        "frequency": "daily",
        "use_gte": True,
        "date_columns": ["filingdate", "transactiondate"],
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
        
        rows = data.get("datatable", {}).get("data", [])
        
        if not rows:
            return False, local_max, 0
        
        # Find max date from returned rows
        api_max = None
        for row in rows:
            if row[0]:
                row_date = datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
                if api_max is None or row_date > api_max:
                    api_max = row_date
        
        if api_max is None:
            return False, local_max, 0
        
        if local_max is None:
            return True, api_max, -1
        
        return True, api_max, -1
        
    except Exception as e:
        logger.warning(f"  Error checking API for {table_name}: {e}")
        return False, None, 0


def download_new_data_paginated(
    table_name: str,
    table_config: Dict,
    since_date: Optional[date],
    api_key: str,
    output_dir: str
) -> Optional[str]:
    """
    Download new data from API with PAGINATION support.
    Returns path to downloaded parquet file, or None if failed.
    """
    date_field = table_config["date_field"]
    use_gte = table_config.get("use_gte", False)
    date_columns = table_config.get("date_columns", [])
    
    # Build base URL
    base_url = f"{BASE_URL}/{table_name}.csv?"
    
    if since_date:
        if use_gte:
            base_url += f"{date_field}.gte={since_date.strftime('%Y-%m-%d')}&"
        else:
            next_date = since_date + timedelta(days=1)
            base_url += f"{date_field}.gte={next_date.strftime('%Y-%m-%d')}&"
    
    base_url += f"api_key={api_key}"
    
    logger.info(f"  Downloading {table_name} data (with pagination)...")
    
    all_dfs = []
    cursor_id = None
    page = 1
    total_rows = 0
    
    try:
        while True:
            # Build URL with cursor if we have one
            if cursor_id:
                url = f"{base_url}&qopts.cursor_id={cursor_id}"
            else:
                url = base_url
            
            # Download this page
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            
            # Check content type - CSV or JSON error?
            content_type = resp.headers.get('content-type', '')
            
            if 'application/json' in content_type:
                # Might be an error or empty response
                try:
                    json_data = resp.json()
                    if 'datatable' in json_data:
                        # Empty result
                        if not json_data['datatable'].get('data'):
                            break
                except:
                    pass
            
            # Parse CSV
            csv_content = resp.text
            if not csv_content.strip() or csv_content.strip() == '':
                break
            
            # Read CSV from string
            df_page = pd.read_csv(io.StringIO(csv_content), parse_dates=date_columns, low_memory=False)
            
            if df_page.empty:
                break
            
            rows_this_page = len(df_page)
            total_rows += rows_this_page
            all_dfs.append(df_page)
            
            logger.info(f"    Page {page}: {rows_this_page:,} rows (total: {total_rows:,})")
            
            # Check for next cursor in the Link header or response
            # Nasdaq Data Link uses cursor_id in response headers or you need JSON endpoint
            # For CSV, we need to check if we got a full page (10,000 rows)
            if rows_this_page < 10000:
                # Less than full page = no more data
                break
            
            # For CSV endpoint, we need to use JSON to get cursor
            # Switch to JSON to get cursor_id
            json_url = url.replace('.csv?', '.json?')
            json_resp = requests.get(json_url, timeout=60)
            json_resp.raise_for_status()
            json_data = json_resp.json()
            
            cursor_id = json_data.get('meta', {}).get('next_cursor_id')
            
            if not cursor_id:
                break
            
            page += 1
            
            # Safety limit
            if page > 50:
                logger.warning(f"  Reached page limit (50), stopping")
                break
        
        if not all_dfs:
            logger.warning(f"  No data downloaded")
            return None
        
        # Combine all pages
        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"  Total downloaded: {len(df):,} rows across {page} page(s)")
        
        # Save to parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parquet_path = os.path.join(output_dir, f"SHARADAR_{table_name}_sync_{timestamp}.parquet")
        df.to_parquet(parquet_path, index=False)
        
        return parquet_path
        
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        import traceback
        traceback.print_exc()
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
    
    # Download new data WITH PAGINATION
    parquet_path = download_new_data_paginated(
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
        
        local_str = str(r['local_max']) if r['local_max'] else 'None'
        
        logger.info(f"{status_icon} {r['table']:8} ({r['db_table']:12}) | "
                   f"Local: {local_str:12} | "
                   f"Status: {r['status']}")
        
        if r["rows_added"] > 0:
            logger.info(f"  └─ Added {r['rows_added']:,} rows")
    
    logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Smart Sharadar data sync (with pagination)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check all tables for new data and sync
  python scripts/smart_data_sync.py --db data/kairos.duckdb
  
  # Check only (show what would be updated)
  python scripts/smart_data_sync.py --db data/kairos.duckdb --check-only
  
  # Sync only SEP and DAILY
  python scripts/smart_data_sync.py --db data/kairos.duckdb --tables SEP DAILY
  
  # Force re-download even if up to date
  python scripts/smart_data_sync.py --db data/kairos.duckdb --tables SF1 --force
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
    logger.info("SHARADAR DATA SYNC (with pagination)")
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