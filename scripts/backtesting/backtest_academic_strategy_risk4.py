#!/usr/bin/env python3
"""
backtest_academic_strategy_risk4.py

Risk Model v4 — Long-Only, Volatility-Aware, ADV-Filtered, Sector-Capped,
MAX 3% per stock, Vol-Targeted, with Turnover Control.

Changes vs Risk3:
    - Adds turnover smoothing vs previous rebalance weights:
        w_pre = (1 - lambda_tc) * w_old + lambda_tc * w_prop
        turnover = sum(|w_pre - w_old|)
        if turnover > turnover_cap:
            w_new = w_old + (turnover_cap / turnover) * (w_pre - w_old)
        else:
            w_new = w_pre
        normalize w_new

Inputs:
    - feat_matrix with:
        ticker, date, alpha_column, target_column, vol_blend, adv_20, size_z
    - tickers with:
        ticker, sector

Usage example:

    python scripts/backtesting/backtest_academic_strategy_risk4.py \
      --db data/kairos.duckdb \
      --alpha-column alpha_composite_v32b \
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
import os
import subprocess
from scripts.log_backtest_run import log_backtest_result # <-- added
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def safe_z(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score with safe fallback."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq: pd.Series) -> float:
    """Max drawdown on equity curve."""
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> tuple[float, float, float]:
    """Annualized return, vol, Sharpe for periodic returns."""
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


def load_data(con, alpha_col, target_col, start, end) -> pd.DataFrame:
    """Load alpha, target, vol_blend, adv_20, size_z, sector from feat_matrix + tickers."""
    logger.info("Loading feat_matrix + sectors for Risk4...")

    where = [
        f"{alpha_col} IS NOT NULL",
        f"{target_col} IS NOT NULL",
        "vol_blend IS NOT NULL",
        "adv_20 IS NOT NULL",
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
            fm.vol_blend,
            fm.adv_20,
            fm.size_z,
            t.sector
        FROM feat_matrix fm
        LEFT JOIN tickers t USING (ticker)
        WHERE {where_sql}
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates.")
    return df


def run_backtest(
    df: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    target_vol: float,
    adv_thresh: float = 2_000_000.0,
    sector_cap_mult: float = 2.0,
    max_stock_w: float = 0.03,
    lambda_tc: float = 0.20,
    turnover_cap: float = 0.20,
) -> pd.DataFrame:
    """
    Long-only, volatility-aware, ADV-filtered, sector-capped,
    3% max stock weight, vol-targeted, with turnover control vs previous rebalance.
    """

    # Phase 1: basic cleaning
    logger.info("Dropping rows with missing sector...")
    df = df.dropna(subset=["sector"])
    if df.empty:
        logger.warning("No data after sector filter.")
        return pd.DataFrame()

    logger.info(f"Filtering by ADV >= {adv_thresh:,.0f}...")
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        logger.warning("No data after ADV filter.")
        return pd.DataFrame()

    # Phase 2: alpha z-scores and clipping
    logger.info("Computing alpha z-scores and clipping...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    records = []
    w_old = pd.Series(dtype=float)  # previous rebalance weights, indexed by ticker

    for d in rebal_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.copy()

        # Sort by alpha and keep top N
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue

        # Ensure unique tickers before sizing
        picks = picks.drop_duplicates(subset="ticker", keep="first")
        picks = picks.set_index("ticker")

        # Initial proposed weights ∝ alpha_z / vol_blend (positive alpha only)
        w_prop = picks["alpha_z"].clip(lower=0.0)
        if w_prop.sum() == 0:
            continue

        w_prop = w_prop / picks["vol_blend"].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w_prop.sum() == 0:
            continue

        w_prop = w_prop / w_prop.sum()

        picks["weight_prop"] = w_prop

        # Cap per-stock weights
        picks["weight_prop"] = picks["weight_prop"].clip(upper=max_stock_w)
        if picks["weight_prop"].sum() == 0:
            continue
        picks["weight_prop"] /= picks["weight_prop"].sum()

        # Sector caps: max sector weight = sector_cap_mult * equal-weight sector share
        sector_counts = universe.groupby("sector")["ticker"].count()
        total_univ = float(len(universe))
        sector_univ_w = sector_counts / total_univ  # eq-weight universe sector weights

        sector_port_w = picks.groupby("sector")["weight_prop"].sum()
        caps = sector_univ_w * sector_cap_mult
        caps = caps.reindex(sector_port_w.index).fillna(0.0)

        for sec, w_sec in sector_port_w.items():
            cap = caps.get(sec, 0.0)
            if w_sec > cap and cap > 0:
                scale = cap / w_sec
                picks.loc[picks["sector"] == sec, "weight_prop"] *= scale

        # Re-normalize after sector caps
        if picks["weight_prop"].sum() <= 0:
            continue
        picks["weight_prop"] /= picks["weight_prop"].sum()

        # Turnover control vs previous rebalance
        w_prop = picks["weight_prop"].copy()
        # ensure no duplicate tickers
        w_prop = w_prop[~w_prop.index.duplicated(keep="first")]

        if w_old.empty:
            # first rebalance
            w_new = w_prop.copy()
        else:
            # Align indices between old and proposed
            all_tickers = pd.Index(w_old.index).union(w_prop.index).unique()
            w_old_full = w_old.reindex(all_tickers).fillna(0.0)
            w_prop_full = w_prop.reindex(all_tickers).fillna(0.0)

            # Proposed smoothed weights
            w_pre = (1 - lambda_tc) * w_old_full + lambda_tc * w_prop_full

            # Compute turnover
            turnover = float((w_pre - w_old_full).abs().sum())
            if turnover > turnover_cap:
                scale = turnover_cap / turnover
                w_new_full = w_old_full + scale * (w_pre - w_old_full)
            else:
                w_new_full = w_pre

            # Normalize
            if w_new_full.sum() <= 0:
                w_new_full = w_old_full
            w_new_full = w_new_full / w_new_full.sum()

            # Restrict to current picks universe
            w_new = w_new_full.reindex(w_prop.index).fillna(0.0)

        # Ensure uniqueness on w_new before carrying forward
        w_new = w_new[~w_new.index.duplicated(keep="first")]
        w_old = w_new.copy()

        # Compute portfolio return on current date
        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue

        port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        bench_ret = float(universe["target"].mean())

        records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
        })

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        logger.warning("Empty backtest after construction.")
        return bt

    # Vol targeting
    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    logger.info(f"Raw annualized vol (pre-scale): {raw_vol:.2%}")

    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    logger.info(f"Target vol={target_vol:.2%}, scale factor={scale:.4f}")

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float) -> None:
    """Print summary stats for Risk Model v4."""
    if bt.empty:
        logger.warning("Empty bt — nothing to summarize.")
        return

    p = bt["port_ret"]
    b = bt["bench_ret"]
    a = bt["active_ret"]

    port_eq = (1.0 + p).cumprod()
    bench_eq = (1.0 + b).cumprod()

    port_total = port_eq.iloc[-1] - 1.0
    bench_total = bench_eq.iloc[-1] - 1.0

    port_ann, port_vol, port_sharpe = annualize(p, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize(b, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize(a, rebalance_every)

    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)

    print("\n==================== RISK MODEL v4 (LONG ONLY, TURNOVER CTRL) ====================")
    print(f"Target Vol:              {target_vol:.2%}")
    print("--------------------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Ret:    {port_ann:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("--------------------------------------------------------------------------")
    print(f"BENCHMARK Total:         {bench_total:7.2%}")
    print(f"BENCHMARK Annual Ret:    {bench_ann:7.2%}")
    print(f"BENCHMARK Annual Vol:    {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max DD:        {bench_dd:7.2%}")
    print("--------------------------------------------------------------------------")
    print(f"ACTIVE Annual Ret:       {active_ann:7.2%}")
    print(f"ACTIVE Annual Vol:       {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("==========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Risk Model v4 Backtester (Long-Only, Turnover-Controlled).")
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
    # === NEW: LOG METRICS ===
    port_total = (1 + bt["port_ret"]).prod() - 1.0
    port_ann, port_vol, port_sharpe = annualize(bt["port_ret"], args.rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize(bt["bench_ret"], args.rebalance_every)
    port_eq = (1 + bt["port_ret"]).cumprod()
    bench_eq = (1 + bt["bench_ret"]).cumprod()
    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)
    active_ann, active_vol, active_sharpe = annualize(bt["active_ret"], args.rebalance_every)


    run_cmd = " ".join(subprocess.check_output(["ps", "-o", "args=", "-p", str(os.getpid())]).decode().strip().split())
    log_backtest_result(
    db_path=args.db,
    strategy_name=os.path.basename(__file__),
    run_command=run_cmd,
    start_date=args.start_date,
    end_date=args.end_date,
    port_total=port_total,
    port_ann=port_ann,
    port_vol=port_vol,
    port_sharpe=port_sharpe,
    port_dd=port_dd,
    bench_total=bench_eq.iloc[-1] - 1.0,
    bench_ann=bench_ann,
    bench_vol=bench_vol,
    bench_sharpe=bench_sharpe,
    bench_dd=bench_dd,
    active_ann=active_ann,
    active_vol=active_vol,
    active_sharpe=active_sharpe,
    )
    con.close()


if __name__ == "__main__":
    main()
