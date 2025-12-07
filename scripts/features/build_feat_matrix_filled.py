#!/usr/bin/env python3
"""
build_feat_matrix_filled.py – RAM-safe, clean join from _filled tables

Creates `feat_matrix_filled` by joining cleaned, forward/backward-filled versions
of feature tables used for model training. Also writes partitioned Parquet.

Outputs:
    - DuckDB table: feat_matrix_filled
    - Partitioned Parquet dataset: scripts/feature_matrices/feat_matrix_filled_parquet/
"""
import duckdb
import pandas as pd
import argparse
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

FILL_TABLES = [
    "feat_price_action_filled",
    "feat_price_shape_filled",
    "feat_stat_filled",
    "feat_trend_filled",
    "feat_volume_volatility_filled",
    "feat_institutional_academic_filled",
    "feat_value_filled",
    "feat_adv_filled",
    "feat_vol_sizing_filled",
    "feat_beta_filled"
]

TARGET_TABLE = "feat_targets"
PARQUET_DIR = Path("scripts/feature_matrices/feat_matrix_filled_parquet")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB file")
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    logger.info("🧱 Building base table from feat_targets")
    con.execute("DROP TABLE IF EXISTS feat_matrix_filled")
    con.execute("CREATE TABLE feat_matrix_filled AS SELECT * FROM feat_targets")

    for t in FILL_TABLES:
        logger.info(f"🔗 Joining {t} ...")
        cols = con.execute(f"PRAGMA table_info('{t}')").fetchdf()["name"].tolist()
        feat_cols = [c for c in cols if c not in ("ticker", "date")]
        if not feat_cols:
            continue

        select_cols = ", ".join([f"f.{c}" for c in feat_cols])

        con.execute(f"""
            CREATE OR REPLACE TABLE feat_matrix_filled AS
            SELECT b.*, {select_cols}
            FROM feat_matrix_filled b
            LEFT JOIN {t} f USING (ticker, date)
        """)

    logger.info("✅ feat_matrix_filled successfully created.")

    # Export to partitioned Parquet
    logger.info(f"💾 Exporting to partitioned Parquet at {PARQUET_DIR}")
    if PARQUET_DIR.exists():
        shutil.rmtree(PARQUET_DIR)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"""
        COPY (
            SELECT * FROM feat_matrix_filled
        ) TO '{str(PARQUET_DIR)}'
        (FORMAT PARQUET, PARTITION_BY (ticker));
    """)

    logger.info("✔ Parquet export complete.")


if __name__ == "__main__":
    main()
