#!/usr/bin/env python3
"""
build_full_feature_matrix.py – RAM-safe, DuckDB-only, partitioned output (overwrite-safe)

Builds the full academic feature matrix by iteratively LEFT JOINing
feat_* tables onto the base (ticker, date) from sep_base_academic,
restricted to your Option B universe.

Outputs:
    1. DuckDB table: feat_matrix
    2. Partitioned Parquet dataset:
       scripts/feature_matrices/<date>_academic_matrix/
       partitioned by ticker
"""

import duckdb
import pandas as pd
import argparse
import logging
from pathlib import Path
import shutil  # For directory removal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Academic feature tables to include
ACADEMIC_TABLES = [
    "feat_price_action",
    "feat_price_shape",
    "feat_stat",
    "feat_trend",
    "feat_volume_volatility",
    "feat_targets",
    "feat_composite_academic",
    "feat_institutional_academic",
    "feat_composite_long",
    "feat_composite_v3",
    # "feat_composite_v4",
]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    parser.add_argument("--date", required=True, help="Snapshot date (YYYY-MM-DD)")
    parser.add_argument("--universe", required=True, help="Ticker universe CSV")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    # ------------------------------------------------------------------
    # Load universe CSV (small) and register as temp table
    # ------------------------------------------------------------------
    logger.info(f"📂 Loading universe CSV: {args.universe}")
    universe_df = pd.read_csv(args.universe)
    con.register("universe_df", universe_df)

    # Validate that required feat_* tables exist
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    missing = [t for t in ACADEMIC_TABLES if t not in existing]
    if missing:
        raise RuntimeError(f"❌ Missing required feature tables: {missing}")

    logger.info("\n🔍 Using academic feature tables:")
    for t in ACADEMIC_TABLES:
        logger.info(f"   • {t}")

    # ------------------------------------------------------------------
    # STEP 1: Build base temp table __feat_base (ticker, date)
    # ------------------------------------------------------------------
    logger.info("\n🔧 Building temp base academic table...")
    con.execute("DROP TABLE IF EXISTS __feat_base")

    con.execute("""
        CREATE TABLE __feat_base AS
        SELECT
            s.ticker,
            s.date
        FROM sep_base_academic s
        INNER JOIN universe_df u USING (ticker)
        ORDER BY s.ticker, s.date
    """)

    base_rows = con.execute("SELECT COUNT(*) FROM __feat_base").fetchone()[0]
    logger.info(f"✔ Base contains {base_rows:,} rows")

    # ------------------------------------------------------------------
    # STEP 2: Iteratively LEFT JOIN each feature table (inside DuckDB)
    # ------------------------------------------------------------------
    logger.info("\n🔧 Iteratively joining feature tables...")

    for t in ACADEMIC_TABLES:
        logger.info(f"   → Joining {t} ...")

        cols_df = con.execute(f"PRAGMA table_info('{t}')").fetchdf()
        all_cols = cols_df["name"].tolist()

        # Exclude join keys
        feat_cols = [c for c in all_cols if c not in ("ticker", "date")]
        if not feat_cols:
            continue

        select_cols = ", ".join([f"f.{c}" for c in feat_cols])

        con.execute(f"""
            CREATE OR REPLACE TABLE __feat_base AS
            SELECT
                b.*,
                {select_cols}
            FROM __feat_base b
            LEFT JOIN {t} f
            USING (ticker, date)
        """)

    # ------------------------------------------------------------------
    # STEP 3: Save to feat_matrix (DuckDB table)
    # ------------------------------------------------------------------
    logger.info("\n💾 Saving final matrix to DuckDB table feat_matrix...")

    con.execute("DROP TABLE IF EXISTS feat_matrix")
    con.execute("""
        CREATE TABLE feat_matrix AS
        SELECT * FROM __feat_base
    """)

    fm_rows = con.execute("SELECT COUNT(*) FROM feat_matrix").fetchone()[0]
    fm_cols = len(con.execute("PRAGMA table_info('feat_matrix')").fetchdf())
    logger.info(f"✔ feat_matrix created: {fm_rows:,} rows × {fm_cols} columns")

    # ------------------------------------------------------------------
    # STEP 4: Export partitioned Parquet (overwrite-safe)
    # ------------------------------------------------------------------
    out_dir = Path(f"scripts/feature_matrices/{args.date}_academic_matrix")

    # Clean old output directory safely
    if out_dir.exists():
        logger.info(f"🧹 Removing old output directory: {out_dir}")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(out_dir)

    logger.info(f"💾 Writing partitioned Parquet dataset to: {out_dir_str}")

    con.execute(f"""
        COPY (
            SELECT * FROM feat_matrix
        ) TO '{out_dir_str}'
        (FORMAT PARQUET, PARTITION_BY (ticker));
    """)

    logger.info("✔ Partitioned Parquet export complete.")

    # Cleanup temp table
    con.execute("DROP TABLE IF EXISTS __feat_base")

    logger.info("✅ Full feature matrix build complete.")


if __name__ == "__main__":
    main()
