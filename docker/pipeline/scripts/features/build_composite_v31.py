#!/usr/bin/env python3
"""
build_composite_v31.py

Composite v3.1 = CS (short-horizon) + CL v2 (long-horizon)

alpha_composite_v31 = 0.6 * alpha_composite_academic
                    + 0.4 * alpha_CL_v2

Inputs:
    feat_composite_academic   (CS)
    feat_composite_long_v2    (CL v2)

Outputs:
    feat_composite_v31
"""

import argparse
import logging
import duckdb
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def build_v31(con: duckdb.DuckDBPyConnection) -> None:
    logger.info("Building Composite v3.1 (CS + CL v2)...")

    # Load CS
    cs = con.execute("""
        SELECT
            ticker,
            date,
            alpha_composite_eq AS alpha_cs
        FROM feat_composite_academic
    """).fetchdf()

    logger.info(f"Loaded {len(cs):,} CS rows.")

    # Load CL v2
    cl2 = con.execute("""
        SELECT
            ticker,
            date,
            alpha_CL_v2
        FROM feat_composite_long_v2
    """).fetchdf()

    logger.info(f"Loaded {len(cl2):,} CL v2 rows.")

    # Join
    df = cs.merge(cl2, on=["ticker", "date"], how="inner")

    logger.info(f"Joined CS + CL v2: {len(df):,} rows.")

    # Composite v3.1
    df["alpha_composite_v31"] = 0.6 * df["alpha_cs"] + 0.4 * df["alpha_CL_v2"]

    # Save table
    con.execute("DROP TABLE IF EXISTS feat_composite_v31")
    con.register("df_v31", df)
    con.execute("CREATE TABLE feat_composite_v31 AS SELECT * FROM df_v31")

    logger.info(f"feat_composite_v31 created ({len(df):,} rows).")


def main():
    parser = argparse.ArgumentParser(description="Build Composite v3.1")
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_v31(con)
    con.close()


if __name__ == "__main__":
    main()
