import argparse
import logging
import os
import sys
import glob
from datetime import datetime, date
import duckdb # type: ignore

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
    conn = duckdb.connect(db_path)
    return conn


def ensure_sep_base_table(conn: duckdb.DuckDBPyConnection, sample_file: str):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sep_base AS
        SELECT * FROM read_parquet(?)
        LIMIT 0
    """, (sample_file,))
    logger.info("Verified or created sep_base table in DuckDB.")


def get_current_max_date(conn: duckdb.DuckDBPyConnection) -> date | None:
    result = conn.execute("SELECT MAX(date) FROM sep_base").fetchone()
    return result[0] if result and result[0] is not None else None


def merge_daily_file(conn: duckdb.DuckDBPyConnection, parquet_path: str, max_date: date | None):
    logger.info(f"Merging daily file: {parquet_path}")
    filter_clause = f"WHERE date > DATE '{max_date}'" if max_date else ""
    insert_sql = f"""
        INSERT INTO sep_base
        SELECT DISTINCT * FROM read_parquet('{parquet_path}')
        {filter_clause}
    """
    try:
        conn.execute(insert_sql)
        logger.info(f"Inserted rows from {os.path.basename(parquet_path)}")
        os.remove(parquet_path)
        logger.info(f"Deleted daily file: {parquet_path}")
    except Exception as e:
        logger.error(f"Failed to merge '{parquet_path}': {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Merge daily SEP Parquets into sep_base DuckDB table.")
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
    args = parser.parse_args()

    daily_files = find_daily_files(args.daily_dir)
    if not daily_files:
        logger.warning("No daily files found. Exiting without changes.")
        sys.exit(0)

    conn = connect_duckdb(args.update_golden)
    ensure_sep_base_table(conn, daily_files[0])
    max_date = get_current_max_date(conn)
    if isinstance(max_date, datetime):
        max_date = max_date.date()

    logger.info(f"Current max date in sep_base: {max_date or 'None'}")

    to_merge = [
        path for path in daily_files
        if max_date is None or parse_date_from_filename(path) > max_date
    ]
    if not to_merge:
        logger.info("sep_base is already up to date; no new files to merge.")
        sys.exit(0)

    for path in to_merge:
        merge_daily_file(conn, path, max_date)

    logger.info("All new daily files merged successfully.")


if __name__ == "__main__":
    main()
