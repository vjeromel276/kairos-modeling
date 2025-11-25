#!/usr/bin/env python3
"""
backtest_academic_strategy_risk3.py

Risk Model v3 — Long-Only, Volatility-Aware, Sector-Capped, Vol-Targeted.

Uses:
    - alpha_column from feat_matrix (e.g. alpha_composite_v31)
    - ret_5d_f as forward return
    - vol_blend from feat_vol_sizing (via feat_matrix)
    - adv_20, size_z from feat_adv (via feat_matrix)
    - sector from tickers

Steps:
    1. Load feat_matrix + tickers (sector)
    2. Drop missing sector or vol_blend
    3. Cross-sectional z-score alpha per date and clip to ±3
    4. Filter to ADV >= threshold (via adv_20)
    5. For each rebalance date:
        a. rank by alpha_z
        b. take top N
        c. initial weights ∝ alpha_z / vol_blend (long-only)
        d. cap at max 3% per stock
        e. apply sector caps (max 2x equal-weight sector share)
    6. Compute raw portfolio returns (5-day fwd)
    7. Vol-target whole portfolio to target_vol
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def safe_z(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq: pd.Series) -> float:
    run_max = eq.cummax()
    dd = eq / run_max - 1
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> tuple[float, float, float]:
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


def load_data(con, alpha_col, target_col, start, end):
    logger.info("Loading feat_matrix + sectors for Risk3...")

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
) -> pd.DataFrame:

    # Clean missing sector
    logger.info("Dropping rows with missing sector...")
    df = df.dropna(subset=["sector"])
    if df.empty:
        logger.warning("No data after sector filtering.")
        return pd.DataFrame()

    # Filter by liquidity (ADV)
    logger.info(f"Filtering by ADV >= {adv_thresh:,.0f}...")
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        logger.warning("No data after ADV filter.")
        return pd.DataFrame()

    # Alpha z-score & clipping
    logger.info("Computing alpha z-scores and clipping...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    records = []

    for d in rebal_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        # Universe for benchmark & sector weights
        universe = day.copy()

        # Rank by alpha_z
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue

        # Initial weights ∝ alpha_z / vol_blend (long-only, only positive alpha)
        w_raw = picks["alpha_z"].clip(lower=0)
        if w_raw.sum() == 0:
            continue

        # Divide by vol_blend for risk-aware sizing
        w_raw = w_raw / picks["vol_blend"].replace(0, np.nan)
        w_raw = w_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if w_raw.sum() == 0:
            continue

        picks["weight"] = w_raw / w_raw.sum()

        # Apply 3% max per stock
        picks["weight"] = picks["weight"].clip(upper=max_stock_w)
        if picks["weight"].sum() == 0:
            continue
        picks["weight"] /= picks["weight"].sum()

        # Sector caps: cap each sector at (sector_cap_mult * equal-weight sector share)
        sector_counts = universe.groupby("sector")["ticker"].count()
        total_univ = float(len(universe))
        sector_univ_w = sector_counts / total_univ  # eq-weight universe share

        sector_port_w = picks.groupby("sector")["weight"].sum()
        caps = sector_univ_w * sector_cap_mult
        caps = caps.reindex(sector_port_w.index).fillna(0.0)

        for sec, w_sec in sector_port_w.items():
            cap = caps.get(sec, 0.0)
            if w_sec > cap and cap > 0:
                scale = cap / w_sec
                picks.loc[picks["sector"] == sec, "weight"] *= scale

        # Renormalize
        tot_w = picks["weight"].sum()
        if tot_w <= 0:
            continue
        picks["weight"] /= tot_w

        port_ret = float((picks["target"] * picks["weight"]).sum())
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
    logger.info(f"Raw annualized vol: {raw_vol:.2%}")

    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    logger.info(f"Target vol={target_vol:.2%}, scale={scale:.4f}")

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float) -> None:
    if bt.empty:
        logger.warning("Nothing to summarize.")
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

    print("\n==================== RISK MODEL v3 (LONG ONLY) ====================")
    print(f"Target Vol:              {target_vol:.2%}")
    print("------------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Ret:    {port_ann:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("------------------------------------------------------------------")
    print(f"BENCHMARK Total:         {bench_total:7.2%}")
    print(f"BENCHMARK Annual Ret:    {bench_ann:7.2%}")
    print(f"BENCHMARK Annual Vol:    {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max DD:        {bench_dd:7.2%}")
    print("------------------------------------------------------------------")
    print(f"ACTIVE Annual Ret:       {active_ann:7.2%}")
    print(f"ACTIVE Annual Vol:       {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("==================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Risk Model v3 Backtester (Long-Only, Vol-Aware).")
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
