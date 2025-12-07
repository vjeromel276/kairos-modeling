#!/usr/bin/env python3
"""
backtest_academic_strategy_ls.py

Phase 8B — Analytic Long–Short backtester.
Dollar-neutral, beta-neutral, sector-capped, vol-targeted.
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np
import sys
import os
import subprocess
from scripts.log_backtest_run import log_backtest_result # <-- added
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
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


def annualize(returns, period_days=5):
    if returns.empty:
        return 0.0, 0.0, 0.0
    af = 252.0 / period_days
    gross = (1 + returns).prod()
    avg = gross ** (1 / len(returns)) - 1
    ann_ret = (1 + avg) ** af - 1
    ann_vol = returns.std() * np.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    return float(ann_ret), float(ann_vol), float(sharpe)


def apply_sector_caps(w, sectors, sector_univ_w, cap_mult):
    w2 = w.copy()
    sec_port = w2.groupby(sectors).sum()
    caps = (sector_univ_w * cap_mult).reindex(sec_port.index).fillna(0)
    for sec, w_sec in sec_port.items():
        cap = caps.get(sec, 0)
        if cap > 0 and w_sec > cap:
            scale = cap / w_sec
            w2.loc[sectors == sec] *= scale
    s = w2.sum()
    return w2 / s if s > 0 else w * 0


def neutralize_exposures_to_beta(expos, betas, min_names=3):
    """Dollar + beta neutrality via regression."""
    e = expos.copy()
    mask = (e != 0) & betas.notna()
    if mask.sum() < min_names:
        return e
    idx = mask[mask].index
    e_vec = e.loc[idx].values
    b_vec = betas.loc[idx].values
    X = np.column_stack([np.ones_like(b_vec), b_vec])
    try:
        theta, *_ = np.linalg.lstsq(X, e_vec, rcond=None)
    except:
        return e
    fitted = X @ theta
    resid = e_vec - fitted
    e2 = e.copy()
    e2.loc[idx] = resid
    return e2


# =====================================================================
# Data load
# =====================================================================

def load_data(con, alpha_col, beta_col, target_col, start, end):
    logger.info("Loading feat_matrix + sectors...")
    where = [
        f"fm.{alpha_col} IS NOT NULL",
        f"fm.{beta_col} IS NOT NULL",
        f"fm.{target_col} IS NOT NULL",
        "fm.vol_blend IS NOT NULL",
        "fm.adv_20 IS NOT NULL"
    ]
    if start:
        where.append(f"fm.date >= DATE '{start}'")
    if end:
        where.append(f"fm.date <= DATE '{end}'")
    where_sql = " AND ".join(where)

    df = con.execute(f"""
        SELECT
            fm.ticker,
            fm.date,
            fm.{alpha_col}  AS alpha,
            fm.{beta_col}   AS beta,
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
# Main LS Backtest
# =====================================================================

def run_backtest(df,
                 top_n_long,
                 top_n_short,
                 rebalance_every,
                 target_vol,
                 adv_thresh=2_000_000,
                 sector_cap_mult=2.0,
                 max_stock_w=0.03):

    logger.info("Cleaning sector + ADV...")
    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        return pd.DataFrame()

    logger.info("Computing alpha z-scores...")
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z).clip(-3, 3)

    dates = sorted(df["date"].unique())
    reb_dates = dates[::rebalance_every]

    results = []

    for d in reb_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        # -------------------------------------
        # DEDUPE universes BEFORE indexing
        # -------------------------------------
        universe = day.drop_duplicates(subset="ticker", keep="last").set_index("ticker")

        # Universe-level sector weights
        sec_counts = universe.groupby("sector").size()
        sec_univ_w = sec_counts / len(universe)

        # ================= LONGS ================
        longs = (universe
                 .sort_values("alpha_z", ascending=False)
                 .head(top_n_long)
                 .copy())

        # Risk-aware long sizing
        lscore = longs["alpha_z"].clip(lower=0) / longs["vol_blend"]
        lscore = lscore.replace([np.inf, -np.inf], 0).fillna(0)
        if lscore.sum() == 0:
            continue
        lw = (lscore / lscore.sum()).clip(upper=max_stock_w)
        lw = lw / lw.sum()
        lw = lw[~lw.index.duplicated(keep="first")]
        lw = apply_sector_caps(lw, longs["sector"], sec_univ_w, sector_cap_mult)

        # ================= SHORTS ================
        shorts = (universe
                  .sort_values("alpha_z", ascending=True)
                  .head(top_n_short)
                  .copy())

        sscore = (-shorts["alpha_z"]).clip(lower=0) / shorts["vol_blend"]
        sscore = sscore.replace([np.inf, -np.inf], 0).fillna(0)
        if sscore.sum() == 0:
            continue
        sw = (sscore / sscore.sum()).clip(upper=max_stock_w)
        sw = sw / sw.sum()
        sw = sw[~sw.index.duplicated(keep="first")]
        sw = apply_sector_caps(sw, shorts["sector"], sec_univ_w, sector_cap_mult)

        # Remove overlap
        overlap = lw.index.intersection(sw.index)
        if len(overlap) > 0:
            lw = lw.drop(overlap, errors="ignore")
            sw = sw.drop(overlap, errors="ignore")

        if lw.empty or sw.empty:
            continue

        # ================= BUILD EXPOSURE VECTOR ================
        combined = pd.Index(sorted(set(lw.index) | set(sw.index)))
        expos = pd.Series(0.0, index=combined)
        expos.loc[lw.index] = 0.5 * lw.values
        expos.loc[sw.index] = -0.5 * sw.values

        expos = expos[~expos.index.duplicated(keep="first")]

        # Align betas
        betas = universe["beta"].reindex(expos.index)

        # Dollar + beta neutral
        expos_n = neutralize_exposures_to_beta(expos, betas)

        gross = expos_n.abs().sum()
        if gross <= 0:
            continue
        expos_f = expos_n / gross

        # Compute returns
        targets = universe["target"].reindex(expos_f.index).fillna(0)
        port_ret = float((expos_f * targets).sum())
        bench_ret = float(universe["target"].mean())
        beta_port = float((expos_f * betas).sum(skipna=True))

        results.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
            "beta_port_raw": beta_port
        })

    bt = pd.DataFrame(results).sort_values("date")
    if bt.empty:
        return bt

    # ================= VOL TARGET =================
    _, raw_vol, _ = annualize(bt["port_ret_raw"], rebalance_every)
    logger.info(f"Raw vol={raw_vol:.2%}")
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]
    bt["beta_port"] = bt["beta_port_raw"] * scale

    return bt


# =====================================================================
# Summary
# =====================================================================

def summarize(bt, reb_days, target_vol):
    if bt.empty:
        print("No results.")
        return

    p = bt["port_ret"]
    b = bt["bench_ret"]
    a = bt["active_ret"]

    port_eq = (1 + p).cumprod()
    bench_eq = (1 + b).cumprod()

    port_total = port_eq.iloc[-1] - 1
    bench_total = bench_eq.iloc[-1] - 1

    port_ann, port_vol, port_sharpe = annualize(p, reb_days)
    bench_ann, bench_vol, bench_sharpe = annualize(b, reb_days)
    active_ann, active_vol, active_sharpe = annualize(a, reb_days)

    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)

    print("\n==================== ANALYTIC LONG–SHORT (PHASE 8B) ====================")
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
    print("-----------------------------------------------------------------------")
    print(f"Avg portfolio beta:      {bt['beta_port'].mean(): .4f}")
    print(f"Std portfolio beta:      {bt['beta_port'].std(): .4f}")
    print("=======================================================================\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", required=True)
    parser.add_argument("--beta-column", required=True)
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--top-n-long", type=int, default=75)
    parser.add_argument("--top-n-short", type=int, default=75)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--target-vol", type=float, default=0.12)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    df = load_data(
        con,
        alpha_col=args.alpha_column,
        beta_col=args.beta_column,
        target_col=args.target_column,
        start=args.start_date,
        end=args.end_date,
    )
    bt = run_backtest(
        df,
        top_n_long=args.top_n_long,
        top_n_short=args.top_n_short,
        rebalance_every=args.rebalance_every,
        target_vol=args.target_vol,
    )
    summarize(bt, reb_days=args.rebalance_every, target_vol=args.target_vol)
    
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
