#!/usr/bin/env python3
"""
backtest_academic_strategy_ls_opt_save.py

Phase 8C — Optimized Long–Short Backtester WITH saving to DuckDB.

This engine includes:
    - Mean–variance QP optimizer (dollar & beta neutral)
    - Sector caps
    - Per-stock weight caps
    - ADV >= 2M filter
    - Covariance shrinkage
    - Vol targeting
    - Stores results in DuckDB table: backtest_results_ls_opt

Use this script INSTEAD of the original optimizer when you want to run Phase 8D.
"""

import argparse
import logging
import os
import subprocess
from typing import List, Tuple, Optional


import duckdb
import numpy as np
import pandas as pd
from scripts.log_backtest_run import log_backtest_result # <-- added


logging.basicConfig(
level=logging.INFO,
format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Utility Functions
# =====================================================================

def safe_z(series: pd.Series) -> pd.Series:
    """Cross-sectional z-score with safe fallback."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq: pd.Series) -> float:
    """Compute max drawdown of equity curve."""
    run_max = eq.cummax()
    return float((eq / run_max - 1.0).min())


def annualize(returns: pd.Series, period_days: int = 5) -> Tuple[float, float, float]:
    """Annualized return, vol, Sharpe from periodic returns."""
    if returns.empty:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1 + returns).prod()
    avg = gross ** (1 / len(returns)) - 1.0
    ann_ret = (1 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


def apply_sector_caps(w: pd.Series, sectors: pd.Series, sector_univ_w: pd.Series, sector_cap_mult: float) -> pd.Series:
    """Apply sector caps to absolute exposures, preserving gross exposure."""
    if w.empty:
        return w

    gross = float(w.abs().sum())
    if gross == 0:
        return w * 0.0

    w2 = w.copy()
    sec_w = w2.groupby(sectors).sum()
    caps = (sector_univ_w * sector_cap_mult).reindex(sec_w.index).fillna(0.0)

    for sec, w_sec in sec_w.items():
        cap = caps.get(sec, 0)
        if abs(w_sec) > cap > 0:
            scale = cap / abs(w_sec)
            w2.loc[sectors == sec] *= scale

    new_gross = float(w2.abs().sum())
    if new_gross > 0:
        w2 *= gross / new_gross

    return w2


def neutralize_exposures_to_beta(expos: pd.Series, betas: pd.Series, min_names: int = 3) -> pd.Series:
    """Force dollar + beta neutrality via regression residuals."""
    e = expos.copy()
    mask = (e != 0.0) & betas.notna()
    if mask.sum() < min_names:
        return e

    idx = mask[mask].index
    e_vec = e.loc[idx].values
    b_vec = betas.loc[idx].values

    X = np.column_stack([np.ones_like(b_vec), b_vec])
    try:
        theta, *_ = np.linalg.lstsq(X, e_vec, rcond=None)
        resid = e_vec - (X @ theta)
        e.loc[idx] = resid
    except Exception:
        pass

    return e


# =====================================================================
# Data Loading + Covariance Builder
# =====================================================================

def load_alpha_universe(con, alpha_col, beta_col, target_col, start, end) -> pd.DataFrame:
    """Load alpha, beta, target, sector, vol_blend, ADV from feat_matrix."""
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
            fm.{alpha_col} AS alpha,
            fm.{beta_col} AS beta,
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

    return df


def load_price_window(con, tickers: List[str], end_date: pd.Timestamp, lookback: int) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    tickers_sql = ",".join(f"'{t}'" for t in tickers)
    ds = end_date.strftime("%Y-%m-%d")

    df = con.execute(f"""
        SELECT ticker, date, close
        FROM sep_base_academic
        WHERE ticker IN ({tickers_sql})
          AND date <= DATE '{ds}'
          AND date > DATE '{ds}' - INTERVAL {lookback} DAY
        ORDER BY date, ticker
    """).fetchdf()

    return df


def build_covariance(
    con,
    tickers: List[str],
    as_of_date: pd.Timestamp,
    cov_window: int = 60,
    shrink_lambda: float = 0.3,
) -> Tuple[np.ndarray, List[str]]:

    raw_prices = load_price_window(con, tickers, as_of_date, cov_window + 20)
    if raw_prices.empty:
        return np.zeros((0, 0)), []

    prices = (
        raw_prices
        .pivot(index="date", columns="ticker", values="close")
        .sort_index()
        .ffill()
    )

    rets = prices.pct_change().dropna(how="all")
    if len(rets) > cov_window:
        rets = rets.iloc[-cov_window:]

    rets = rets.fillna(0.0)

    tickers_used = list(rets.columns)
    if len(tickers_used) == 0:
        return np.zeros((0, 0)), []

    X = rets.values
    Sigma = np.cov(X, rowvar=False)

    diag = np.diag(np.diag(Sigma))
    Sigma = shrink_lambda * diag + (1 - shrink_lambda) * Sigma

    Sigma += 1e-6 * np.eye(Sigma.shape[0])  # ridge for PD safety

    return Sigma, tickers_used


# =====================================================================
# Mean-Variance QP Solver
# =====================================================================

def solve_qp(Sigma: np.ndarray, alpha: np.ndarray, beta: np.ndarray, risk_aversion: float) -> np.ndarray:
    """Solve MVO with dollar & beta neutrality via KKT system."""
    n = len(alpha)
    if n == 0:
        return np.zeros(0)

    A = np.column_stack([np.ones(n), beta])   # constraints: sum(w)=0, sum(w*beta)=0

    # Build KKT:
    K_top = np.hstack([Sigma, A])
    K_bottom = np.hstack([A.T, np.zeros((2, 2))])
    K = np.vstack([K_top, K_bottom])

    rhs = np.concatenate([alpha / risk_aversion, np.zeros(2)])

    try:
        sol = np.linalg.solve(K, rhs)
        return sol[:n]
    except Exception:
        return alpha / (np.linalg.norm(alpha) + 1e-12)


# =====================================================================
# Optimized LS Backtest
# =====================================================================

def run_backtest(
    con,
    df,
    top_n_long: int,
    top_n_short: int,
    rebalance_every: int,
    target_vol: float,
    adv_thresh: float,
    sector_cap_mult: float,
    max_stock_w: float,
    cov_window: int,
    shrink_lambda: float,
    risk_aversion: float,
):
    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z).clip(-3, 3)

    dates = sorted(df["date"].unique())
    reb_dates = dates[::rebalance_every]

    out = []

    for d in reb_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.drop_duplicates(subset="ticker", keep="last").set_index("ticker")

        # long/short candidates
        longs = universe.sort_values("alpha_z", ascending=False).head(top_n_long)
        shorts = universe.sort_values("alpha_z", ascending=True).head(top_n_short)

        cand = sorted(set(longs.index) | set(shorts.index))
        if not cand:
            continue

        Sigma, cov_ticks = build_covariance(
            con, cand, pd.to_datetime(d), cov_window, shrink_lambda
        )
        if len(cov_ticks) == 0:
            continue

        sub = universe.reindex(cov_ticks)
        alpha = sub["alpha_z"].values
        beta = sub["beta"].fillna(0.0).values

        w_raw = solve_qp(Sigma, alpha, beta, risk_aversion)
        w = pd.Series(w_raw, index=cov_ticks)

        # enforce sign consistency with alpha
        w = w.where(sub["alpha_z"] > 0, 0.0) + w.where(sub["alpha_z"] < 0, 0.0)

        # abs cap
        w = w.clip(-max_stock_w, max_stock_w)

        # sector caps
        sec_counts = universe.groupby("sector").size()
        sec_univ_w = sec_counts / float(len(universe))
        w = apply_sector_caps(w, sub["sector"], sec_univ_w, sector_cap_mult)

        # normalize gross=1
        gross = float(w.abs().sum())
        if gross > 0:
            w = w / gross

        # final neutralization
        bet = sub["beta"].fillna(0.0)
        w = neutralize_exposures_to_beta(w, bet)

        # renorm
        gross2 = float(w.abs().sum())
        if gross2 > 0:
            w = w / gross2

        # compute return
        targets = sub["target"].reindex(w.index).fillna(0.0)
        port_ret = float((w * targets).sum())
        bench_ret = float(universe["target"].mean())
        beta_port = float((w * bet).sum())

        out.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
            "beta_port_raw": beta_port,
        })

    bt = pd.DataFrame(out).sort_values("date")

    if bt.empty:
        return bt

    # vol targeting
    _, raw_vol, _ = annualize(bt["port_ret_raw"], rebalance_every)
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]
    bt["beta_port"] = bt["beta_port_raw"] * scale

    return bt


# =====================================================================
# Summary
# =====================================================================

def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float):
    if bt.empty:
        print("No results.")
        return

    p = bt["port_ret"]
    b = bt["bench_ret"]

    port_eq = (1 + p).cumprod()
    bench_eq = (1 + b).cumprod()

    port_total = port_eq.iloc[-1] - 1.0
    _, port_vol, port_sharpe = annualize(p, rebalance_every)
    _, bench_vol, bench_sharpe = annualize(b, rebalance_every)

    port_dd = max_drawdown(port_eq)

    print("\n==================== OPTIMIZED LONG–SHORT (PHASE 8C) ====================")
    print(f"Target Vol:              {target_vol:.2%}")
    print("-----------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("=================================================================\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 8C Optimized LS Backtester with Save.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", required=True)
    parser.add_argument("--beta-column", required=True)
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--top-n-long", type=int, default=150)
    parser.add_argument("--top-n-short", type=int, default=150)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--target-vol", type=float, default=0.10)
    parser.add_argument("--adv-thresh", type=float, default=2_000_000.0)
    parser.add_argument("--sector-cap-mult", type=float, default=2.0)
    parser.add_argument("--max-stock-w", type=float, default=0.03)
    parser.add_argument("--cov-window", type=int, default=60)
    parser.add_argument("--shrink-lambda", type=float, default=0.3)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    args = parser.parse_args()
    con = duckdb.connect(args.db)

    df = load_alpha_universe(
        con,
        alpha_col=args.alpha_column,
        beta_col=args.beta_column,
        target_col=args.target_column,
        start=args.start_date,
        end=args.end_date,
    )

    bt = run_backtest(
        con,
        df,
        top_n_long=args.top_n_long,
        top_n_short=args.top_n_short,
        rebalance_every=args.rebalance_every,
        target_vol=args.target_vol,
        adv_thresh=args.adv_thresh,
        sector_cap_mult=args.sector_cap_mult,
        max_stock_w=args.max_stock_w,
        cov_window=args.cov_window,
        shrink_lambda=args.shrink_lambda,
        risk_aversion=args.risk_aversion,
    )

    summarize(bt, rebalance_every=args.rebalance_every, target_vol=args.target_vol)

    # *** SAVE RESULTS ***
    con.execute("DROP TABLE IF EXISTS backtest_results_ls_opt")
    con.register("bt_ls_opt", bt)
    con.execute("CREATE TABLE backtest_results_ls_opt AS SELECT * FROM bt_ls_opt")
    logger.info("Saved backtest_results_ls_opt to DuckDB.")

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
