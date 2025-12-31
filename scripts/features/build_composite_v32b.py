#!/usr/bin/env python3
"""
build_composite_v32b.py

Composite v3.2b (Balanced Horizon Blend):

    alpha_composite_v32b = 0.4 * CS
                         + 0.3 * CL2
                         + 0.3 * SMOOTHED

Where:
    CS    = alpha_composite_eq       (feat_composite_academic)
    CL2   = alpha_CL_v2              (feat_composite_long_v2)
    SMOOTH = alpha_smoothed_v31      (feat_alpha_smoothed_v31)

Outputs:
    feat_composite_v32b
"""

import argparse
import logging
import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def build_v32b(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Loading CS (feat_composite_academic)...")
    cs = con.execute("""
        SELECT
            ticker,
            date,
            alpha_composite_eq AS alpha_cs
        FROM feat_composite_academic
    """).fetchdf()
    logger.info(f"CS rows: {len(cs):,}")

    logger.info("Loading CL v2 (feat_composite_long_v2)...")
    cl2 = con.execute("""
        SELECT
            ticker,
            date,
            alpha_CL_v2
        FROM feat_composite_long_v2
    """).fetchdf()
    logger.info(f"CL2 rows: {len(cl2):,}")

    logger.info("Loading smoothed alpha (feat_alpha_smoothed_v31)...")
    sm = con.execute("""
        SELECT
            ticker,
            date,
            alpha_smoothed
        FROM feat_alpha_smoothed_v31
    """).fetchdf()
    logger.info(f"SM rows: {len(sm):,}")

    # Left join on CS (primary universe)
    logger.info("Joining CS + CL2 + SM on (ticker, date)...")
    df = cs.merge(cl2, on=["ticker", "date"], how="left")
    df = df.merge(sm,  on=["ticker", "date"], how="left")
    logger.info(f"Joined rows: {len(df):,}")

    # Compute balanced blend: 0.4 CS, 0.3 CL2, 0.3 SM
    df["alpha_composite_v32b"] = (
          0.4 * df["alpha_cs"]
        + 0.3 * df["alpha_CL_v2"].fillna(0.0)
        + 0.3 * df["alpha_smoothed"].fillna(0.0)
    )

    # Enforce uniqueness
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"])
    after = len(df)
    logger.info(f"Dropped {before - after:,} duplicate rows. Final rows: {after:,}")

    con.execute("DROP TABLE IF EXISTS feat_composite_v32b")
    con.register("df_v32b", df)
    con.execute("CREATE TABLE feat_composite_v32b AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM df_v32b")

    logger.info("feat_composite_v32b created successfully.")


def main():
    parser = argparse.ArgumentParser(description="Build Composite v3.2b (Balanced Horizon).")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_v32b(con)
    con.close()


if __name__ == "__main__":
    main()
