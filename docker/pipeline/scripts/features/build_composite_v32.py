#!/usr/bin/env python3
"""
build_composite_v32.py

Composite v3.2 = IC-weighted blend of three components:
    - CS   : alpha_composite_eq      (short-horizon composite)
    - CL2  : alpha_CL_v2             (long-horizon composite v2)
    - SM   : alpha_smoothed_v31      (smoothed version of v3.1)

alpha_v32 = w_cs * CS + w_cl2 * CL2 + w_sm * SM

Weights (w_cs, w_cl2, w_sm) are based on historical daily cross-sectional IC
with 5-day forward returns, computed from CLEAN feature tables (not feat_matrix).

Output:
    feat_composite_v32 with exactly one row per (ticker, date).
"""

import argparse
import logging
import duckdb  # type: ignore
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_ic_weights(con: duckdb.DuckDBPyConnection) -> tuple[float, float, float]:
    """
    Compute IC-based weights for CS, CL2, SM using CLEAN tables:
      - feat_targets             -> ret_5d_f
      - feat_composite_academic  -> alpha_composite_eq (CS)
      - feat_composite_long_v2   -> alpha_CL_v2 (CL2)
      - feat_alpha_smoothed_v31  -> alpha_smoothed (SM)
    """

    logger.info("Loading alpha + targets for IC computation (clean tables, not feat_matrix)...")

    df = con.execute("""
        SELECT
            t.date,
            t.ticker,
            a.alpha_composite_eq AS alpha_cs,
            l.alpha_CL_v2        AS alpha_cl2,
            s.alpha_smoothed     AS alpha_sm,
            t.ret_5d_f           AS target
        FROM feat_targets t
        LEFT JOIN feat_composite_academic   a ON a.ticker = t.ticker AND a.date = t.date
        LEFT JOIN feat_composite_long_v2    l ON l.ticker = t.ticker AND l.date = t.date
        LEFT JOIN feat_alpha_smoothed_v31   s ON s.ticker = t.ticker AND s.date = t.date
        WHERE a.alpha_composite_eq IS NOT NULL
          AND l.alpha_CL_v2        IS NOT NULL
          AND s.alpha_smoothed     IS NOT NULL
          AND t.ret_5d_f           IS NOT NULL
        ORDER BY t.date, t.ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows for IC calc.")

    # Compute daily cross-sectional IC for each component
    def daily_ic(col: str) -> float:
        ics = df.groupby("date").apply(
            lambda x: x[col].corr(x["target"])
        )
        return float(ics.mean())

    ic_cs = daily_ic("alpha_cs")
    ic_cl2 = daily_ic("alpha_cl2")
    ic_sm = daily_ic("alpha_sm")

    logger.info(f"IC(cs)={ic_cs:.6f}, IC(cl2)={ic_cl2:.6f}, IC(sm)={ic_sm:.6f}")

    ic_vals = np.array([ic_cs, ic_cl2, ic_sm])
    ic_vals = np.nan_to_num(ic_vals)

    # Shift so smallest IC is zero, then normalize by sum
    ic_vals = ic_vals - ic_vals.min()
    if ic_vals.sum() == 0:
        logger.warning("All ICs are ~0; defaulting to equal weights.")
        return 1/3, 1/3, 1/3

    weights = ic_vals / ic_vals.sum()
    w_cs, w_cl2, w_sm = float(weights[0]), float(weights[1]), float(weights[2])
    logger.info(f"Using IC-based weights: w_cs={w_cs:.4f}, w_cl2={w_cl2:.4f}, w_sm={w_sm:.4f}")
    return w_cs, w_cl2, w_sm


def build_v32(con: duckdb.DuckDBPyConnection) -> None:
    """
    Build feat_composite_v32 **only** from clean feature tables:
      - feat_composite_academic   (CS)
      - feat_composite_long_v2    (CL2)
      - feat_alpha_smoothed_v31   (SM)

    We left-join on CS (alpha_composite_academic) as the primary key,
    and ensure exactly one row per (ticker, date).
    """

    w_cs, w_cl2, w_sm = compute_ic_weights(con)

    logger.info("Building Composite v3.2 from clean features (no feat_matrix)...")

    cs = con.execute("""
        SELECT
            ticker,
            date,
            alpha_composite_eq AS alpha_cs
        FROM feat_composite_academic
    """).fetchdf()

    logger.info(f"CS rows: {len(cs):,}")

    cl2 = con.execute("""
        SELECT
            ticker,
            date,
            alpha_CL_v2
        FROM feat_composite_long_v2
    """).fetchdf()

    logger.info(f"CL2 rows: {len(cl2):,}")

    sm = con.execute("""
        SELECT
            ticker,
            date,
            alpha_smoothed
        FROM feat_alpha_smoothed_v31
    """).fetchdf()

    logger.info(f"SM rows: {len(sm):,}")

    # Left join on CS (primary universe)
    df = cs.merge(cl2, on=["ticker", "date"], how="left")
    df = df.merge(sm,  on=["ticker", "date"], how="left")

    logger.info(f"Joined CS+CL2+SM rows: {len(df):,}")

    # Compute weighted blend with rescaling for missing components
    comp_cols = ["alpha_cs", "alpha_CL_v2", "alpha_smoothed"]
    weights = np.array([w_cs, w_cl2, w_sm])

    vals = df[comp_cols].to_numpy()
    mask = ~np.isnan(vals)
    weighted = np.where(mask, vals * weights, 0.0)
    weight_sum = (mask * weights).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha_v32 = np.divide(weighted.sum(axis=1), weight_sum, where=weight_sum > 0)
    alpha_v32 = np.nan_to_num(alpha_v32, nan=0.0)

    df["alpha_composite_v32"] = alpha_v32

    # Enforce uniqueness: one row per (ticker, date)
    before = len(df)
    df = df.drop_duplicates(subset=["ticker", "date"])
    after = len(df)
    logger.info(f"Dropped {before - after:,} duplicate (ticker,date) rows. Final rows: {after:,}")

    con.execute("DROP TABLE IF EXISTS feat_composite_v32")
    con.register("df_v32", df)
    con.execute("CREATE TABLE feat_composite_v32 AS SELECT * FROM df_v32")

    logger.info("feat_composite_v32 created successfully with 1 row per (ticker,date).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    build_v32(con)
    con.close()


if __name__ == "__main__":
    main()
