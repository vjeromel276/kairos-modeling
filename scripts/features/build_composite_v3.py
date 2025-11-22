#!/usr/bin/env python3
"""
build_composite_v3.py

Composite v3: multi-horizon blend of:
    - CS (short composite):  alpha_composite_eq
    - CL (long composite):   alpha_CL

Formula:
    alpha_composite_v3 = 0.6 * alpha_composite_eq + 0.4 * alpha_CL

Inputs:
    - feat_composite_academic  (from build_academic_composite_factors.py)
    - feat_composite_long      (from build_composite_long.py)

Output:
    - feat_composite_v3 (DuckDB table)
"""

import argparse
import logging
import duckdb  # type: ignore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_composite_v3(con: duckdb.DuckDBPyConnection) -> None:
    required = ["feat_composite_academic", "feat_composite_long"]
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    missing = [t for t in required if t not in existing]
    if missing:
        raise RuntimeError(f"Missing required tables for Composite v3: {missing}")

    logger.info("Building Composite v3 (CS + CL blend)...")

    con.execute("DROP TABLE IF EXISTS feat_composite_v3")

    # Blend CS and CL on (ticker, date)
    con.execute("""
        CREATE TABLE feat_composite_v3 AS
        SELECT
            a.ticker,
            a.date,
            a.alpha_composite_eq,
            l.alpha_CL,
            0.6 * a.alpha_composite_eq + 0.4 * l.alpha_CL AS alpha_composite_v3
        FROM feat_composite_academic a
        JOIN feat_composite_long l
          USING (ticker, date)
    """)

    count = con.execute("SELECT COUNT(*) FROM feat_composite_v3").fetchone()[0]
    logger.info(f"feat_composite_v3 created with {count:,} rows.")


def main():
    parser = argparse.ArgumentParser(description="Build Composite v3 (CS + CL blend).")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_composite_v3(con)
    con.close()


if __name__ == "__main__":
    main()
