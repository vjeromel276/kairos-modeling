#!/usr/bin/env python3
"""
backtest_academic_strategy_risk4_with_save.py

Phase 8A — Long-only, volatility-aware, ADV-filtered, sector-capped,
max-weight capped, vol-targeted, with turnover control.

This version is IDENTICAL to your original backtest_academic_strategy_risk4.py
(:contentReference[oaicite:0]{index=0}) but adds one final step:

    Saves the backtest results into DuckDB table:
        backtest_results_longonly_r4

This allows Phase 8D (regime switching) to load long-only returns cleanly.

No contract changes. No logic changes. Just persistence added.
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


# =====================================================================
# Utility helpers
# =====================================================================

def safe_z(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq):
    run_max = eq.cummax()
    return float((eq / run_max - 1).min())


def annualize(returns: pd.Series, period_days: int = 5):
    if returns.empty:
        return 0.0, 0.0, 0.0
    af = 252.0 / period_days
    gross = (1 + returns).prod()
    avg = gross ** (1 / len(returns)) - 1
    ann_ret = (1 + avg) ** af - 1
    ann_vol = returns.std() * np.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    return float(ann_ret), float(ann_vol), float(sharpe)


# =====================================================================
# Data loader
# =====================================================================

def load_data(con, alpha_col, target_col, start, end):
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
        LEFT JOIN tickers t USING(ticker)
        WHERE {where_sql}
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates.")
    return df


# =====================================================================
# Core backtest
# =====================================================================

def run_backtest(
    df,
    top_n,
    rebalance_every,
    target_vol,
    adv_thresh=2_000_000.0,
    sector_cap_mult=2.0,
    max_stock_w=0.03,
    lambda_tc=0.20,
    turnover_cap=0.20,
):

    logger.info("Dropping missing sectors...")
    df = df.dropna(subset=["sector"])
    if df.empty:
        return pd.DataFrame()

    logger.info(f"Filtering ADV >= {adv_thresh:,.0f}...")
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        return pd.DataFrame()

    logger.info("Z-scoring alphas...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z).clip(-3, 3)

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    w_old = pd.Series(dtype=float)
    recs = []

    for d in rebal_dates:
        today = df[df["date"] == d].copy()
        if today.empty:
            continue

        universe = today.copy()

        picks = today.sort_values("alpha_z", ascending=False).head(top_n).copy()
        picks = picks.drop_duplicates(subset="ticker").set_index("ticker")
        if picks.empty:
            continue

        w_prop = picks["alpha_z"].clip(lower=0)
        w_prop = w_prop / picks["vol_blend"].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / w_prop.sum()

        w_prop = w_prop.clip(upper=max_stock_w)
        w_prop = w_prop / w_prop.sum()

        sector_counts = universe.groupby("sector")["ticker"].count()
        sec_univ_w = sector_counts / float(len(universe))

        sector_port = picks.groupby("sector")["alpha"].size()
        caps = sec_univ_w * sector_cap_mult

        for sec, _ in picks.groupby("sector"):
            cap = caps.get(sec, 0)
            w_sec = w_prop[picks["sector"] == sec].sum()
            if cap > 0 and w_sec > cap:
                scale = cap / w_sec
                tickers = picks[picks["sector"] == sec].index
                w_prop.loc[tickers] *= scale

        if w_prop.sum() <= 0:
            continue
        w_prop = w_prop / w_prop.sum()

        if w_old.empty:
            w_new = w_prop.copy()
        else:
            all_tk = w_old.index.union(w_prop.index)
            w_old_f = w_old.reindex(all_tk).fillna(0)
            w_prop_f = w_prop.reindex(all_tk).fillna(0)

            w_pre = (1 - lambda_tc) * w_old_f + lambda_tc * w_prop_f
            turnover = float((w_pre - w_old_f).abs().sum())
            if turnover > turnover_cap:
                scale = turnover_cap / turnover
                w_new_f = w_old_f + scale * (w_pre - w_old_f)
            else:
                w_new_f = w_pre

            if w_new_f.sum() <= 0:
                w_new_f = w_old_f
            w_new_f = w_new_f / w_new_f.sum()
            w_new = w_new_f.reindex(w_prop.index).fillna(0)

        w_old = w_new.copy()

        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue

        port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        bench_ret = float(universe["target"].mean())

        recs.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
        })

    bt = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt

    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    logger.info(f"Raw vol={raw_vol:.2%}")
    scale = target_vol / raw_vol if raw_vol > 0 else 1

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


# =====================================================================
# Summary
# =====================================================================

def summarize(bt, rebalance_every, target_vol):
    if bt.empty:
        print("Nothing to summarize.")
        return
    p, b = bt["port_ret"], bt["bench_ret"]
    eq = (1 + p).cumprod()
    bench_eq = (1 + b).cumprod()
    port_total = float(eq.iloc[-1] - 1)
    bench_total = float(bench_eq.iloc[-1] - 1)
    ann_ret, ann_vol, ann_sharpe = annualize(p, rebalance_every)
    bench_ret, bench_vol, bench_sharpe = annualize(b, rebalance_every)
    print("\n==================== RISK4 (Long Only) ====================")
    print(f"PORT Total Return: {port_total:7.2%}")
    print(f"PORT Annual Ret:   {ann_ret:7.2%}")
    print(f"PORT Vol:          {ann_vol:7.2%}")
    print(f"PORT Sharpe:       {ann_sharpe:7.2f}")
    print(f"PORT Max DD:       {max_drawdown(eq):7.2%}")
    print("-----------------------------------------------------------")
    print(f"BENCH Total:       {bench_total:7.2%}")
    print(f"BENCH Annual Ret:  {bench_ret:7.2%}")
    print(f"BENCH Vol:         {bench_vol:7.2%}")
    print(f"BENCH Sharpe:      {bench_sharpe:7.2f}")
    print("===========================================================\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Risk Model v4 (Long Only) + Save")
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

    summarize(bt, args.rebalance_every, args.target_vol)

    # NEW: save results for 8D
    con.execute("DROP TABLE IF EXISTS backtest_results_longonly_r4")
    con.register("bt_r4", bt)
    con.execute("CREATE TABLE backtest_results_longonly_r4 AS SELECT * FROM bt_r4")
    logger.info("Saved backtest_results_longonly_r4 to DuckDB.")
    
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
