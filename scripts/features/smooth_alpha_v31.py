#!/usr/bin/env python3
"""
smooth_alpha_v31.py

Applies exponential smoothing to alpha_composite_v31.

alpha_smoothed = 0.7 * alpha_composite_v31 + 0.3 * lag(alpha_composite_v31)

Output:
    feat_alpha_smoothed_v31
"""

import argparse
import logging
import duckdb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def build_smoothed(con):
    logger.info("Loading alpha_composite_v31...")
    df = con.execute("""
        SELECT ticker, date, alpha_composite_v31
        FROM feat_composite_v31
        ORDER BY ticker, date
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for smoothing.")

    def smooth(group):
        group["alpha_smoothed"] = group["alpha_composite_v31"].ewm(
            alpha=0.3, adjust=False
        ).mean()
        return group

    df = df.groupby("ticker", group_keys=False).apply(smooth)

    con.execute("DROP TABLE IF EXISTS feat_alpha_smoothed_v31")
    con.register("df_sm", df)
    con.execute("CREATE TABLE feat_alpha_smoothed_v31 AS SELECT * REPLACE (CAST(date AS DATE) AS date) FROM df_sm")
    logger.info("Created feat_alpha_smoothed_v31.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_smoothed(con)
    con.close()


if __name__ == "__main__":
    main()
