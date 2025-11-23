#!/usr/bin/env python3
"""
build_composite_v4.py — DEDUPE-SAFE, IC-WEIGHTED

This script:
    1. Computes daily cross-sectional IC for CS & CL
    2. Averages IC to compute stable weights
    3. Builds feat_composite_v4 safely using DISTINCT
"""

import duckdb
import pandas as pd
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def compute_ic_weights(con, start, end):
    logger.info("Computing IC weights...")

    where = [
        "alpha_composite_eq IS NOT NULL",
        "alpha_CL IS NOT NULL",
        "ret_5d_f IS NOT NULL",
    ]
    if start:
        where.append(f"date >= DATE '{start}'")
    if end:
        where.append(f"date <= DATE '{end}'")

    where_sql = " AND ".join(where)

    df = con.execute(f"""
        SELECT
            date,
            corr(alpha_composite_eq, ret_5d_f) AS ic_cs,
            corr(alpha_CL,          ret_5d_f) AS ic_cl
        FROM feat_matrix
        WHERE {where_sql}
        GROUP BY date
        HAVING COUNT(*) > 20
        ORDER BY date
    """).fetchdf()

    if df.empty:
        raise RuntimeError("No IC samples found!")

    ic_cs = df["ic_cs"].mean()
    ic_cl = df["ic_cl"].mean()

    logger.info(f"Mean IC(CS) = {ic_cs:.6f}")
    logger.info(f"Mean IC(CL) = {ic_cl:.6f}")

    denom = abs(ic_cs) + abs(ic_cl)
    if denom == 0:
        logger.warning("ICs both 0 → default to 0.5/0.5")
        return 0.5, 0.5

    w_cs = ic_cs / denom
    w_cl = ic_cl / denom

    logger.info(f"IC weights → CS: {w_cs:.4f}, CL: {w_cl:.4f}")
    return w_cs, w_cl


def build_v4(con, start=None, end=None):
    logger.info("Building dedupe-safe Composite v4...")

    w_cs, w_cl = compute_ic_weights(con, start, end)

    con.execute("DROP TABLE IF EXISTS feat_composite_v4")

    con.execute(f"""
        CREATE TABLE feat_composite_v4 AS
        SELECT DISTINCT
            ticker,
            date,
            alpha_composite_eq,
            alpha_CL,
            ({w_cs}) * alpha_composite_eq +
            ({w_cl}) * alpha_CL AS alpha_composite_v4
        FROM feat_matrix
        WHERE alpha_composite_eq IS NOT NULL
          AND alpha_CL IS NOT NULL
    """)

    count = con.execute("SELECT COUNT(*) FROM feat_composite_v4").fetchone()[0]
    logger.info(f"feat_composite_v4 created, rows = {count:,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_v4(con, args.train_start, args.train_end)
    con.close()


if __name__ == "__main__":
    main()
