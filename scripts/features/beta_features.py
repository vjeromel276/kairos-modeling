#!/usr/bin/env python3
"""
beta_features.py

Builds rolling **market beta** features for each (ticker, date) in
`sep_base_academic`, using an equal-weighted "market" made from the
Option B universe itself.

Definitions
-----------
- ret_1d:        daily close-to-close return per ticker
- mkt_ret_1d:    equal-weight daily market return across all tickers
- beta_21d:      rolling 21-day CAPM-style beta vs mkt_ret_1d
- beta_63d:      rolling 63-day CAPM-style beta vs mkt_ret_1d
- beta_252d:     rolling 252-day CAPM-style beta vs mkt_ret_1d
- resid_vol_63d: rolling 63-day std.dev of idiosyncratic residual
                 (ret_1d - beta_63d * mkt_ret_1d)

Output
------
Creates DuckDB table: feat_beta

Columns:
    ticker
    date
    beta_21d
    beta_63d
    beta_252d
    resid_vol_63d

Usage:
    python scripts/features/beta_features.py --db data/kairos.duckdb
"""

import argparse
import logging

import duckdb  # type: ignore
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_beta_features(con: duckdb.DuckDBPyConnection) -> None:
    """
    Compute rolling market beta features vs an equal-weighted market
    made from sep_base_academic itself.
    """

    logger.info("Loading close prices from sep_base_academic for beta calc...")

    df = con.execute(
        """
        SELECT
            ticker,
            date,
            close
        FROM sep_base_academic
        ORDER BY ticker, date
        """
    ).fetchdf()

    logger.info(f"Loaded {len(df):,} rows from sep_base_academic.")

    # ------------------------------------------------------------------
    # 1) Per-ticker daily returns
    # ------------------------------------------------------------------
    logger.info("Computing per-ticker daily returns ret_1d...")
    df["ret_1d"] = df.groupby("ticker")["close"].pct_change()

    # ------------------------------------------------------------------
    # 2) Equal-weight market return (across all tickers) per date
    # ------------------------------------------------------------------
    logger.info("Computing equal-weighted market return mkt_ret_1d...")
    mkt = (
        df.groupby("date")["ret_1d"]
        .mean()  # NaNs from first-day-per-ticker are ignored by default
        .rename("mkt_ret_1d")
    )

    df = df.join(mkt, on="date")

    # ------------------------------------------------------------------
    # 3) Rolling betas by ticker
    # ------------------------------------------------------------------
    logger.info("Computing rolling betas (21d / 63d / 252d) per ticker...")

    def add_betas(g: pd.DataFrame) -> pd.DataFrame:
        r = g["ret_1d"]
        m = g["mkt_ret_1d"]

        # Helper to avoid division-by-zero var(m)
        def beta_rolling(window: int) -> pd.Series:
            cov = r.rolling(window).cov(m)
            var = m.rolling(window).var()
            beta = cov / var.replace(0.0, np.nan)
            return beta

        g["beta_21d"] = beta_rolling(21)
        g["beta_63d"] = beta_rolling(63)
        g["beta_252d"] = beta_rolling(252)

        # Idiosyncratic residual vol vs 63d beta
        resid = r - g["beta_63d"] * m
        g["resid_vol_63d"] = resid.rolling(63).std()

        return g

    df = df.groupby("ticker", group_keys=False).apply(add_betas)

    # ------------------------------------------------------------------
    # 4) Clean up and persist
    # ------------------------------------------------------------------
    # We at least require a valid 63d beta to use in long/short beta-neutrality.
    logger.info("Dropping rows without beta_63d (warm-up period)...")
    before = len(df)
    df = df.dropna(subset=["beta_63d"])
    after = len(df)
    logger.info(f"Kept {after:,} rows (dropped {before - after:,} warm-up rows).")

    out = df[
        [
            "ticker",
            "date",
            "beta_21d",
            "beta_63d",
            "beta_252d",
            "resid_vol_63d",
        ]
    ]

    logger.info("Writing feat_beta to DuckDB...")
    con.execute("DROP TABLE IF EXISTS feat_beta")
    con.register("df_beta", out)
    con.execute("CREATE TABLE feat_beta AS SELECT * FROM df_beta")
    logger.info(f"feat_beta created with {len(out):,} rows.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rolling beta features.")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    try:
        build_beta_features(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()
