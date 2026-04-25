#!/usr/bin/env python3
"""
scripts/refresh_duckdb_v2.py
============================
Refresh any Sharadar table from the Nasdaq Data Link API. Designed as a
one-stop tool for a full Sharadar refresh; for the daily production loop,
prefer `smart_data_sync_v2.py` (smaller surface area, faster).

Tables handled:
  SEP        - Daily stock prices (raw mirror of SHARADAR/SEP)     [incremental]
  DAILY      - Daily fundamental ratios                            [incremental]
  SF1        - Quarterly/Annual fundamentals                       [incremental]
  SF2        - Insider transactions                                [incremental]
  SF3        - Institutional holdings (long form)                  [incremental]
  SF3A       - Institutional holdings by ticker                    [incremental]
  SF3B       - Institutional holdings by investor                  [incremental]
  SFP        - Sharadar Fund Prices                                [incremental]
  METRICS    - Daily snapshot metrics (52w hi/lo, MAs, betas)      [incremental]
  TICKERS    - Ticker metadata (sector, industry, listings)        [full reload]
  ACTIONS    - Corporate actions (splits, dividends)               [incremental]
  EVENTS     - Corporate event filings                             [incremental]
  SP500      - S&P 500 constituent changes                         [full reload]
  INDICATORS - Indicator metadata reference (no date field)        [full reload, opt-in]

Full reload is used where deletions matter (TICKERS) or where the table is a
small change-log/reference that's cheap to re-download (SP500, INDICATORS).
INDICATORS has no date column and barely changes, so it's opt-in only — you
must list it explicitly via --tables.

Note: SEP here targets the legacy `sep` table. The production `sep_base` table
is maintained by smart_data_sync_v2.py (different db_table name, same source).

Usage:
    # Refresh everything that's stale (excluding INDICATORS)
    python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb

    # Check only (no download)
    python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --check-only

    # Refresh specific tables
    python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --tables METRICS TICKERS

    # Refresh INDICATORS reference table (must be explicit)
    python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --tables INDICATORS

    # Force re-download even if up to date
    python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --tables SF3 --force

Environment:
    NASDAQ_DATA_LINK_API_KEY: Your Nasdaq Data Link API key
"""

import argparse
import io
import logging
import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional, Dict, List, Tuple

import duckdb
import pandas as pd
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

API_KEY_ENV = "NASDAQ_DATA_LINK_API_KEY"
BASE_URL = "https://data.nasdaq.com/api/v3/datatables/SHARADAR"
PAGE_SAFETY_LIMIT = 500  # ~5M rows max per table per run

TABLES: Dict[str, Dict] = {
    "SEP": {
        "db_table": "sep",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Daily stock prices (raw)",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["date", "lastupdated"],
    },
    "DAILY": {
        "db_table": "daily",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Daily fundamental ratios (PE, PB, PS, EV/EBITDA)",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["date", "lastupdated"],
    },
    "SF1": {
        "db_table": "sf1",
        "date_field": "lastupdated",
        "db_date_field": "lastupdated",
        "description": "Quarterly/Annual fundamentals",
        "reload_mode": "incremental",
        "use_gte": True,
        "date_columns": ["datekey", "reportperiod", "lastupdated"],
    },
    "SF2": {
        "db_table": "sf2",
        "date_field": "filingdate",
        "db_date_field": "filingdate",
        "description": "Insider transactions",
        "reload_mode": "incremental",
        "use_gte": True,
        "date_columns": ["filingdate", "transactiondate"],
    },
    "SFP": {
        "db_table": "sfp",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Sharadar Fund Prices (mutual funds, ETFs)",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["date", "lastupdated"],
    },
    "METRICS": {
        "db_table": "metrics",
        "date_field": "lastupdated",
        "db_date_field": "lastupdated",
        "description": "Daily snapshot metrics (52w hi/lo, MAs, betas)",
        "reload_mode": "incremental",
        "use_gte": True,
        "date_columns": ["date", "lastupdated"],
    },
    "TICKERS": {
        "db_table": "tickers",
        "date_field": "lastupdated",
        "db_date_field": "lastupdated",
        "description": "Ticker metadata (sector, industry, listings)",
        "reload_mode": "full",
        "date_columns": [
            "lastupdated", "firstadded", "firstpricedate", "lastpricedate",
            "firstquarter", "lastquarter",
        ],
    },
    "ACTIONS": {
        "db_table": "actions",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Corporate actions (splits, dividends)",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["date"],
    },
    "EVENTS": {
        "db_table": "events",
        "date_field": "date",
        "db_date_field": "date",
        "description": "Corporate event filings",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["date"],
    },
    "SF3": {
        "db_table": "sf3",
        "date_field": "calendardate",
        "db_date_field": "calendardate",
        "description": "Institutional holdings (long form)",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["calendardate"],
    },
    "SF3A": {
        "db_table": "sf3a",
        "date_field": "calendardate",
        "db_date_field": "calendardate",
        "description": "Institutional holdings by ticker",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["calendardate"],
    },
    "SF3B": {
        "db_table": "sf3b",
        "date_field": "calendardate",
        "db_date_field": "calendardate",
        "description": "Institutional holdings by investor",
        "reload_mode": "incremental",
        "use_gte": False,
        "date_columns": ["calendardate"],
    },
    "SP500": {
        "db_table": "sp500",
        "date_field": "date",
        "db_date_field": "date",
        "description": "S&P 500 constituent changes",
        "reload_mode": "full",
        "date_columns": ["date"],
    },
    "INDICATORS": {
        "db_table": "indicators",
        "description": "Indicator metadata reference (no date field)",
        "reload_mode": "full",
        "no_date_field": True,
        "opt_in_only": True,
        "date_columns": [],
    },
}

DEFAULT_TABLES = [t for t, cfg in TABLES.items() if not cfg.get("opt_in_only")]


def get_api_key() -> str:
    api_key = os.getenv(API_KEY_ENV)
    if not api_key:
        logger.error(f"Environment variable '{API_KEY_ENV}' not set.")
        sys.exit(1)
    return api_key


def get_local_max_date(conn: duckdb.DuckDBPyConnection, table_config: Dict) -> Optional[date]:
    db_table = table_config["db_table"]
    date_field = table_config["db_date_field"]

    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    if db_table not in tables:
        logger.info(f"  Table '{db_table}' does not exist locally")
        return None

    try:
        result = conn.execute(f"SELECT MAX({date_field}) FROM {db_table}").fetchone()
        if result and result[0]:
            v = result[0]
            if isinstance(v, datetime):
                return v.date()
            if isinstance(v, date):
                return v
            return datetime.strptime(str(v)[:10], "%Y-%m-%d").date()
        return None
    except Exception as e:
        logger.warning(f"  Error reading max({date_field}) from {db_table}: {e}")
        return None


def check_api_for_new_data(
    table_name: str,
    table_config: Dict,
    local_max: Optional[date],
    api_key: str,
) -> Tuple[bool, Optional[date]]:
    """Returns (has_new_data, api_max_date_observed)."""
    date_field = table_config["date_field"]
    url = f"{BASE_URL}/{table_name}.json?"

    if local_max:
        check_date = local_max + timedelta(days=1)
        url += f"{date_field}.gte={check_date.strftime('%Y-%m-%d')}&"

    url += f"qopts.columns={date_field}&"
    url += f"qopts.per_page=100&"
    url += f"api_key={api_key}"

    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("datatable", {}).get("data", [])
        if not rows:
            return False, local_max

        api_max = None
        for row in rows:
            if not row[0]:
                continue
            d = datetime.strptime(str(row[0])[:10], "%Y-%m-%d").date()
            if api_max is None or d > api_max:
                api_max = d
        if api_max is None:
            return False, local_max
        return True, api_max
    except Exception as e:
        logger.warning(f"  Error checking API for {table_name}: {e}")
        return False, None


def download_paginated(
    table_name: str,
    table_config: Dict,
    since_date: Optional[date],
    api_key: str,
) -> Optional[pd.DataFrame]:
    """Download from API with pagination. since_date=None means full table."""
    date_field = table_config["date_field"]
    use_gte = table_config.get("use_gte", False)
    date_columns = table_config.get("date_columns", [])

    base_url = f"{BASE_URL}/{table_name}.csv?"
    if since_date is not None:
        if use_gte:
            base_url += f"{date_field}.gte={since_date.strftime('%Y-%m-%d')}&"
        else:
            next_date = since_date + timedelta(days=1)
            base_url += f"{date_field}.gte={next_date.strftime('%Y-%m-%d')}&"
    base_url += f"api_key={api_key}"

    logger.info(f"  Downloading {table_name} (paginated)...")

    all_dfs: List[pd.DataFrame] = []
    cursor_id: Optional[str] = None
    page = 1
    total_rows = 0

    try:
        while True:
            url = f"{base_url}&qopts.cursor_id={cursor_id}" if cursor_id else base_url
            resp = requests.get(url, timeout=180)
            resp.raise_for_status()

            csv_content = resp.text
            if not csv_content.strip():
                break

            df_page = pd.read_csv(
                io.StringIO(csv_content),
                parse_dates=date_columns,
                low_memory=False,
            )
            if df_page.empty:
                break

            rows_this = len(df_page)
            total_rows += rows_this
            all_dfs.append(df_page)
            logger.info(f"    Page {page}: {rows_this:,} rows (total: {total_rows:,})")

            if rows_this < 10000:
                break

            # Get next cursor via JSON endpoint
            json_url = url.replace(".csv?", ".json?")
            json_resp = requests.get(json_url, timeout=60)
            json_resp.raise_for_status()
            cursor_id = json_resp.json().get("meta", {}).get("next_cursor_id")
            if not cursor_id:
                break

            page += 1
            if page > PAGE_SAFETY_LIMIT:
                logger.warning(f"  Reached page limit ({PAGE_SAFETY_LIMIT}), stopping")
                break

        if not all_dfs:
            logger.warning(f"  No data downloaded")
            return None

        df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"  Total: {len(df):,} rows across {page} page(s)")
        return df
    except Exception as e:
        logger.error(f"  Download failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def insert_incremental(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_config: Dict,
) -> int:
    """Append only rows strictly newer than local max. Returns rows added."""
    db_table = table_config["db_table"]
    date_field = table_config["db_date_field"]

    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    if db_table not in tables:
        logger.info(f"  Creating table '{db_table}'...")
        conn.execute(f"CREATE TABLE {db_table} AS SELECT * FROM df")
        return conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]

    local_max = get_local_max_date(conn, table_config)
    before = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
    filter_clause = f"WHERE {date_field} > '{local_max}'" if local_max else ""
    conn.execute(f"INSERT INTO {db_table} SELECT DISTINCT * FROM df {filter_clause}")
    after = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
    return after - before


def replace_full(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    table_config: Dict,
) -> Tuple[int, int]:
    """Drop and recreate the table from df. Returns (rows_before, rows_after)."""
    db_table = table_config["db_table"]
    tables = conn.execute("SHOW TABLES").fetchdf()["name"].tolist()
    before = (
        conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
        if db_table in tables else 0
    )
    conn.execute(f"DROP TABLE IF EXISTS {db_table}")
    conn.execute(f"CREATE TABLE {db_table} AS SELECT * FROM df")
    after = conn.execute(f"SELECT COUNT(*) FROM {db_table}").fetchone()[0]
    return before, after


def refresh_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    table_config: Dict,
    api_key: str,
    check_only: bool,
    force: bool,
) -> Dict:
    mode = table_config["reload_mode"]
    result = {
        "table": table_name,
        "db_table": table_config["db_table"],
        "mode": mode,
        "local_max": None,
        "api_max": None,
        "rows_before": None,
        "rows_after": None,
        "rows_added": 0,
        "status": "unknown",
    }

    logger.info(f"\n{'='*60}")
    logger.info(f"{table_name} [{mode}]: {table_config['description']}")
    logger.info(f"{'='*60}")

    no_date = table_config.get("no_date_field", False)
    if no_date:
        # Reference table with no date column — always treat as needing reload.
        logger.info(f"  No date field; will full-reload on demand")
        local_max = None
        api_max = None
        has_new = True
    else:
        local_max = get_local_max_date(conn, table_config)
        logger.info(f"  Local max {table_config['db_date_field']}: {local_max or 'No data'}")
        has_new, api_max = check_api_for_new_data(table_name, table_config, local_max, api_key)
        if api_max:
            logger.info(f"  API max {table_config['date_field']}: {api_max}")
    result["local_max"] = local_max
    result["api_max"] = api_max

    if not has_new and not force:
        if api_max is None:
            logger.info(f"  ? Could not determine API status")
            result["status"] = "check_failed"
        else:
            logger.info(f"  + Already up to date")
            result["status"] = "up_to_date"
        return result

    if force:
        logger.info(f"  Force download requested")
    else:
        logger.info(f"  New data available")

    if check_only:
        logger.info(f"  (check-only mode, skipping download)")
        result["status"] = "needs_update"
        return result

    # For full reload, download the entire table (since_date=None).
    # For incremental, download from local_max forward.
    since = None if mode == "full" else local_max
    df = download_paginated(table_name, table_config, since, api_key)

    if df is None:
        result["status"] = "download_failed"
        return result

    if mode == "full":
        before, after = replace_full(conn, df, table_config)
        result["rows_before"] = before
        result["rows_after"] = after
        result["rows_added"] = after - before
        logger.info(f"  + Replaced table: {before:,} -> {after:,} rows ({after - before:+,})")
    else:
        added = insert_incremental(conn, df, table_config)
        result["rows_added"] = added
        result["rows_after"] = conn.execute(
            f"SELECT COUNT(*) FROM {table_config['db_table']}"
        ).fetchone()[0]
        new_max = get_local_max_date(conn, table_config)
        result["local_max"] = new_max
        logger.info(f"  + Added {added:,} rows, new max: {new_max}")

    result["status"] = "updated"
    return result


def print_summary(results: List[Dict]):
    logger.info(f"\n{'='*70}")
    logger.info("REFRESH SUMMARY")
    logger.info(f"{'='*70}")
    icons = {
        "up_to_date": "+",
        "updated": "+",
        "needs_update": "!",
        "download_failed": "X",
        "check_failed": "?",
    }
    for r in results:
        icon = icons.get(r["status"], "?")
        local_str = str(r["local_max"]) if r["local_max"] else "None"
        logger.info(
            f"{icon} {r['table']:8} [{r['mode']:11}] | "
            f"Local: {local_str:12} | Status: {r['status']}"
        )
        if r["mode"] == "full" and r["rows_before"] is not None:
            logger.info(
                f"    rows: {r['rows_before']:,} -> {r['rows_after']:,} "
                f"({r['rows_added']:+,})"
            )
        elif r["rows_added"]:
            logger.info(f"    +{r['rows_added']:,} rows")
    logger.info(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Refresh slow-moving Sharadar tables (METRICS, TICKERS, ACTIONS, EVENTS, SF3*, SP500)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh all stale tables
  python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb

  # Check only (show what would update)
  python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --check-only

  # Refresh just metrics and tickers
  python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --tables METRICS TICKERS

  # Force re-download (ignore staleness check)
  python scripts/refresh_duckdb_v2.py --db data/kairos.duckdb --tables SF3 --force
""",
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLES.keys()),
        default=DEFAULT_TABLES,
        help=f"Tables to refresh (default: {', '.join(DEFAULT_TABLES)}; "
             f"opt-in only: {', '.join(t for t in TABLES if t not in DEFAULT_TABLES)})",
    )
    parser.add_argument("--check-only", action="store_true", help="Only check, don't download")
    parser.add_argument("--force", action="store_true", help="Force download even if up to date")

    args = parser.parse_args()

    api_key = get_api_key()

    logger.info(f"\n{'='*70}")
    logger.info("SHARADAR SLOW-TABLE REFRESH v2")
    logger.info(f"{'='*70}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Tables: {', '.join(args.tables)}")
    logger.info(f"Mode: {'Check only' if args.check_only else 'Refresh'}")

    conn = duckdb.connect(args.db)

    results: List[Dict] = []
    for table_name in args.tables:
        result = refresh_table(
            conn=conn,
            table_name=table_name,
            table_config=TABLES[table_name],
            api_key=api_key,
            check_only=args.check_only,
            force=args.force,
        )
        results.append(result)

    conn.close()

    print_summary(results)

    failed = any(r["status"] in ("download_failed", "check_failed") for r in results)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
