#!/usr/bin/env python3
"""
backtest_academic_strategy_risk2.py

Risk-Controlled Backtester v2A (LONG-ONLY)

Adds on top of v1:
    - Cross-sectional z-scoring and winsorization of alpha
    - Alpha-weighted long-only (no shorts)
    - 3% max single-stock weight
    - Sector caps (max 2x sector's universe weight)
    - Volatility targeting to desired annual volatility (e.g. 20%)

Intentionally DOES NOT:
    - Use volume/ADV yet
    - Do full sector neutrality (only caps)
    - Do beta neutrality (that's for long-short mode later)

Inputs:
    - feat_matrix table with:
        ticker, date, alpha_column, target_column
    - tickers table with:
        ticker, sector

Usage example:

    python scripts/backtesting/backtest_academic_strategy_risk2.py \
        --db data/kairos.duckdb \
        --alpha-column alpha_composite_v3 \
        --target-column ret_5d_f \
        --top-n 75 \
        --rebalance-every 5 \
        --target-vol 0.20 \
        --start-date 2015-01-01 \
        --end-date 2025-11-01
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------- Utility functions ----------

def safe_z(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(equity: pd.Series) -> float:
    run_max = equity.cummax()
    dd = equity / run_max - 1
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> tuple[float, float, float]:
    """Annualized stats for periodic returns."""
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


# ---------- Data loading ----------

def load_data(
    con: duckdb.DuckDBPyConnection,
    alpha_col: str,
    target_col: str,
    start: str | None,
    end: str | None,
) -> pd.DataFrame:
    """
    Load alpha + target + sector from feat_matrix, joined with tickers.

    NOTE: We only pull columns we KNOW exist:
      - ticker, date, alpha, target, sector
    """
    logger.info("Loading feat_matrix ...")

    where = [
        f"{alpha_col} IS NOT NULL",
        f"{target_col} IS NOT NULL",
    ]
    if start:
        where.append(f"date >= DATE '{start}'")
    if end:
        where.append(f"date <= DATE '{end}'")

    where_sql = " AND ".join(where)

    df = con.execute(f"""
        SELECT
            fm.ticker,
            fm.date,
            fm.{alpha_col} AS alpha,
            fm.{target_col} AS target,
            t.sector
        FROM feat_matrix fm
        LEFT JOIN tickers t USING (ticker)
        WHERE {where_sql}
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates.")
    return df


# ---------- Core backtest ----------

def run_backtest(
    df: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    target_vol: float,
    sector_cap_multiplier: float = 2.0,
    max_stock_weight: float = 0.03,
) -> pd.DataFrame:
    """
    LONG-ONLY risk-controlled backtest:

      - Drop rows with missing sector
      - Z-score alpha and clip to ±3 per date
      - For each rebalance date:
          * select top-N by alpha_z
          * initial alpha-weighted long-only weights (no negative)
          * cap each stock at max_stock_weight (e.g. 3%)
          * enforce sector caps: sector_weight <= sector_cap_multiplier * sector_universe_weight
      - Volatility targeting to target_vol

    Returns:
      DataFrame with port_ret, bench_ret, active_ret time series.
    """

    # ---- Clean missing sectors ----
    logger.info("Dropping rows with missing sector ...")
    df = df.dropna(subset=["sector"])

    if df.empty:
        logger.warning("No data after sector cleanup.")
        return pd.DataFrame()

    # ---- Cross-sectional alpha z-score & clipping ----
    logger.info("Computing alpha z-scores and clipping ...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    dates = sorted(df["date"].unique())
    rebalance_dates = dates[::rebalance_every]

    records = []

    for d in rebalance_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        # Universe for benchmark & sector weights
        universe = day

        # Rank by alpha_z
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue

        # ---- Initial alpha-weighted weights (long-only) ----
        raw_w = picks["alpha_z"].clip(lower=0)
        if raw_w.sum() == 0:
            continue
        raw_w = raw_w / raw_w.sum()

        picks["weight"] = raw_w

        # ---- 3% max single-stock weight ----
        picks["weight"] = picks["weight"].clip(upper=max_stock_weight)
        picks["weight"] = picks["weight"] / picks["weight"].sum()

        # ---- Sector caps: max 2x sector universe weight ----
        # Universe sector weights (equal-weight across universe)
        sector_counts = universe.groupby("sector")["ticker"].count()
        total_universe = float(len(universe))
        sector_univ_w = sector_counts / total_universe  # equal-weight universe

        # Portfolio sector weights from current picks
        sector_port_w = picks.groupby("sector")["weight"].sum()

        # For any sector where port > cap, scale down proportionally
        caps = sector_univ_w * sector_cap_multiplier
        caps = caps.reindex(sector_port_w.index).fillna(0.0)

        # Scale sectors exceeding cap
        for sec, w_sec in sector_port_w.items():
            cap = caps.get(sec, 0.0)
            if w_sec > cap and cap > 0:
                scale = cap / w_sec
                picks.loc[picks["sector"] == sec, "weight"] *= scale

        # Renormalize after sector caps
        total_w = picks["weight"].sum()
        if total_w <= 0:
            continue
        picks["weight"] /= total_w

        # ---- Portfolio return (5-day forward) ----
        port_ret = float((picks["target"] * picks["weight"]).sum())
        bench_ret = float(universe["target"].mean())

        records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
        })

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        logger.warning("Backtest result is empty.")
        return bt

    # ---- Vol targeting ----
    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    logger.info(f"Raw annualized vol: {raw_vol:.2%}")

    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    logger.info(f"Target vol={target_vol:.2%}, scale factor={scale:.4f}")

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


# ---------- Summary ----------

def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float) -> None:
    """Print summary stats for long-only risk-controlled backtest."""

    if bt.empty:
        logger.warning("Empty bt DataFrame; nothing to summarize.")
        return

    p = bt["port_ret"]
    b = bt["bench_ret"]
    a = bt["active_ret"]

    port_eq = (1.0 + p).cumprod()
    bench_eq = (1.0 + b).cumprod()

    port_total = float(port_eq.iloc[-1] - 1.0)
    bench_total = float(bench_eq.iloc[-1] - 1.0)

    port_ann, port_vol, port_sharpe = annualize(p, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize(b, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize(a, rebalance_every)

    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)

    print("\n==================== RISK-CONTROLLED v2A (LONG ONLY) ====================")
    print(f"Target Vol:              {target_vol:.2%}")
    print("-----------------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Ret:    {port_ann:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("-----------------------------------------------------------------------")
    print(f"BENCHMARK Total:         {bench_total:7.2%}")
    print(f"BENCHMARK Annual Ret:    {bench_ann:7.2%}")
    print(f"BENCHMARK Annual Vol:    {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max DD:        {bench_dd:7.2%}")
    print("-----------------------------------------------------------------------")
    print(f"ACTIVE Annual Ret:       {active_ann:7.2%}")
    print(f"ACTIVE Annual Vol:       {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("=======================================================================\n")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Risk-Control v2A Backtester (Long-Only).")
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", required=True)
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--top-n", type=int, default=75)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--target-vol", type=float, default=0.20)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    df = load_data(
        con,
        alpha_col=args.alpha_column,
        target_col=args.target_column,
        start=args.start_date,
        end=args.end_date,
    )

    bt = run_backtest(
        df,
        top_n=args.top_n,
        rebalance_every=args.rebalance_every,
        target_vol=args.target_vol,
    )

    summarize(bt, rebalance_every=args.rebalance_every, target_vol=args.target_vol)

    con.close()


if __name__ == "__main__":
    main()
