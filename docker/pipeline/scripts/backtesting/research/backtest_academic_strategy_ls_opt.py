#!/usr/bin/env python3
"""
backtest_academic_strategy_ls_opt.py

Phase 8C — Optimized Long–Short Backtester

Mean–variance optimizer with:
    - Dollar neutrality: sum(w) = 0
    - Beta neutrality:   sum(w * beta) = 0
    - ADV filter (default: >= 2,000,000)
    - Per-name caps and sector caps (applied post-optimization)
    - Vol targeting to desired annual volatility (e.g. 10% or 12%)

Covariance is built ON THE FLY from sep_base_academic:
    - Rolling window of past daily returns (e.g. 60 days)
    - Shrinkage toward diagonal for stability

Usage example:

    python scripts/backtesting/backtest_academic_strategy_ls_opt.py \
      --db data/kairos.duckdb \
      --alpha-column alpha_composite_v33_regime \
      --beta-column beta_252d \
      --target-column ret_5d_f \
      --top-n-long 150 \
      --top-n-short 150 \
      --rebalance-every 5 \
      --target-vol 0.10 \
      --start-date 2015-01-01 \
      --end-date 2025-11-28
"""

import argparse
import logging
from typing import List, Tuple, Optional

import duckdb
import numpy as np
import pandas as pd

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


def max_drawdown(eq: pd.Series) -> float:
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> Tuple[float, float, float]:
    if returns.empty:
        return 0.0, 0.0, 0.0
    af = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** af - 1.0
    ann_vol = returns.std() * np.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


def apply_sector_caps(
    w: pd.Series,
    sectors: pd.Series,
    sector_univ_w: pd.Series,
    sector_cap_mult: float,
) -> pd.Series:
    """
    Apply per-side sector caps, then renormalize to preserve sum(|w|).
    """
    if w.empty:
        return w

    gross = float(w.abs().sum())
    if gross <= 0:
        return w * 0.0

    # Work on absolute weights per side
    w_side = w.copy()

    sec_w = w_side.groupby(sectors).sum()
    caps = (sector_univ_w * sector_cap_mult).reindex(sec_w.index).fillna(0.0)

    for sec, w_sec in sec_w.items():
        cap = caps.get(sec, 0.0)
        if cap > 0 and abs(w_sec) > cap:
            scale = cap / abs(w_sec)
            tickers = sectors[sectors == sec].index
            w_side.loc[tickers] *= scale

    # Renormalize back to original gross exposure
    new_gross = float(w_side.abs().sum())
    if new_gross > 0:
        w_side *= gross / new_gross
    return w_side


def neutralize_exposures_to_beta(
    expos: pd.Series,
    betas: pd.Series,
    min_names: int = 3,
) -> pd.Series:
    """
    Dollar + beta neutrality via regression:
        e_raw = a * 1 + b * beta + e_resid
    We subtract the fitted component, keeping only e_resid.
    """
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
    except Exception:
        return e

    fitted = X @ theta
    resid = e_vec - fitted

    e2 = e.copy()
    e2.loc[idx] = resid
    return e2


# =====================================================================
# Data loading
# =====================================================================

def load_alpha_universe(
    con: duckdb.DuckDBPyConnection,
    alpha_col: str,
    beta_col: str,
    target_col: str,
    start: Optional[str],
    end: Optional[str],
) -> pd.DataFrame:
    """
    Load feat_matrix + tickers with alpha, beta, target, vol_blend, adv_20, sector.
    """
    logger.info("Loading feat_matrix + sectors for LS optimization...")

    where = [
        f"fm.{alpha_col} IS NOT NULL",
        f"fm.{beta_col} IS NOT NULL",
        f"fm.{target_col} IS NOT NULL",
        "fm.vol_blend IS NOT NULL",
        "fm.adv_20 IS NOT NULL",
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
        LEFT JOIN tickers t USING (ticker)
        WHERE {where_sql}
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows across {df['date'].nunique()} dates.")
    return df


def load_price_window(
    con: duckdb.DuckDBPyConnection,
    tickers: List[str],
    as_of_date: pd.Timestamp,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """
    Load price history for tickers for covariance estimation window.
    """
    if not tickers:
        return pd.DataFrame()

    tickers_list = ",".join(f"'{t}'" for t in tickers)
    date_str = as_of_date.strftime("%Y-%m-%d")

    df = con.execute(f"""
        SELECT
            ticker,
            date,
            close
        FROM sep_base_academic
        WHERE ticker IN ({tickers_list})
          AND date <= DATE '{date_str}'
          AND date > DATE '{date_str}' - INTERVAL {lookback_days} DAY
        ORDER BY date, ticker
    """).fetchdf()

    return df


def build_covariance(
    con: duckdb.DuckDBPyConnection,
    tickers: List[str],
    as_of_date: pd.Timestamp,
    cov_window: int = 60,
    extra_buffer: int = 30,
    shrink_lambda: float = 0.3,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a shrunk covariance matrix Σ for the given tickers at as_of_date.

    Returns:
        Sigma (n x n), tickers_used (list)
    """
    if not tickers:
        return np.zeros((0, 0)), []

    raw_prices = load_price_window(
        con,
        tickers,
        as_of_date,
        lookback_days=cov_window + extra_buffer,
    )
    if raw_prices.empty:
        return np.zeros((0, 0)), []

    prices = (raw_prices
              .pivot(index="date", columns="ticker", values="close")
              .sort_index()
              .ffill())

    returns = prices.pct_change().dropna(how="all")
    if len(returns) < cov_window // 2:
        # Not enough data, use identity covariance
        tickers_used = list(returns.columns)
        n = len(tickers_used)
        return np.eye(n), tickers_used

    # Use most recent cov_window rows
    if len(returns) > cov_window:
        returns = returns.iloc[-cov_window:]

    # Drop columns with all NaN
    returns = returns.dropna(axis=1, how="all")
    tickers_used = list(returns.columns)
    if len(tickers_used) == 0:
        return np.zeros((0, 0)), []

    # Fill remaining NaNs with 0 (neutral for covariance)
    returns = returns.fillna(0.0)

    # Sample covariance
    X = returns.values  # shape (T, n)
    Sigma = np.cov(X, rowvar=False)

    # Shrinkage toward diagonal
    diag = np.diag(np.diag(Sigma))
    Sigma_shrunk = shrink_lambda * diag + (1.0 - shrink_lambda) * Sigma

    # Ensure positive definiteness by adding small ridge if needed
    eps = 1e-6
    Sigma_shrunk += eps * np.eye(Sigma_shrunk.shape[0])

    return Sigma_shrunk, tickers_used


# =====================================================================
# Optimization
# =====================================================================

def solve_mean_variance_qp(
    Sigma: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Solve:

        minimize  (1/2) w^T Sigma w - (1/risk_aversion) alpha^T w
        subject to:
            sum(w)      = 0      (dollar neutral)
            sum(w*beta) = 0      (beta neutral)

    via KKT system:

        [Sigma   A] [w]   = [alpha / risk_aversion]
        [A^T    0] [λ]     [         0            ]

    where A has columns: 1, beta.
    """
    n = Sigma.shape[0]
    if n == 0:
        return np.zeros(0)

    # Build constraint matrix A: (n x 2)
    A = np.column_stack([np.ones(n), beta])

    # KKT matrix
    K_top = np.hstack([Sigma, A])
    K_bottom = np.hstack([A.T, np.zeros((2, 2))])
    K = np.vstack([K_top, K_bottom])

    rhs = np.concatenate([alpha / risk_aversion, np.zeros(2)])

    try:
        sol = np.linalg.solve(K, rhs)
        w = sol[:n]
    except np.linalg.LinAlgError:
        # Fallback: simple scaled alpha if KKT fails
        w = alpha / (np.linalg.norm(alpha) + 1e-8)

    return w


# =====================================================================
# Core backtest
# =====================================================================

def run_optimized_ls_backtest(
    con: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    top_n_long: int,
    top_n_short: int,
    rebalance_every: int,
    target_vol: float,
    adv_thresh: float = 2_000_000.0,
    sector_cap_mult: float = 2.0,
    max_stock_w: float = 0.03,
    cov_window: int = 60,
    shrink_lambda: float = 0.3,
    risk_aversion: float = 1.0,
) -> pd.DataFrame:

    logger.info("Starting optimized long–short backtest (Phase 8C)...")

    # Clean and filter
    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        logger.warning("Empty universe after ADV/sector filters.")
        return pd.DataFrame()

    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z).clip(-3.0, 3.0)

    dates = sorted(df["date"].unique())
    reb_dates = dates[::rebalance_every]

    records = []

    for d in reb_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        # Dedupe per-day universe on ticker
        universe = day.drop_duplicates(subset="ticker", keep="last").set_index("ticker")

        # Compute equal-weight sector shares for caps
        sector_counts = universe.groupby("sector").size()
        sector_univ_w = sector_counts / float(len(universe))

        # Long candidates: top N by alpha_z
        longs = (universe
                 .sort_values("alpha_z", ascending=False)
                 .head(top_n_long)
                 .copy())

        # Short candidates: bottom N by alpha_z
        shorts = (universe
                  .sort_values("alpha_z", ascending=True)
                  .head(top_n_short)
                  .copy())

        if longs.empty or shorts.empty:
            continue

        # Candidate tickers for optimization
        cand_tickers = sorted(set(longs.index) | set(shorts.index))

        # Build covariance for candidate set
        Sigma, cov_tickers = build_covariance(
            con,
            cand_tickers,
            pd.to_datetime(d),
            cov_window=cov_window,
            shrink_lambda=shrink_lambda,
        )
        if len(cov_tickers) == 0:
            continue

        # Align alpha and beta
        sub = universe.reindex(cov_tickers)
        alpha_vec = sub["alpha_z"].values
        beta_vec = sub["beta"].fillna(0.0).values

        # Solve QP for raw weights
        w_raw = solve_mean_variance_qp(Sigma, alpha_vec, beta_vec, risk_aversion=risk_aversion)

        w = pd.Series(w_raw, index=cov_tickers)

        # Enforce that longs/shorts follow alpha sign loosely:
        # zero out weights where alpha_z sign conflicts strongly
        long_mask = sub["alpha_z"] > 0
        short_mask = sub["alpha_z"] < 0
        # optional: allow small cross sign, but we keep it simple:
        w = w.where(long_mask | short_mask, 0.0)

        # Per-name cap in absolute weight
        w = w.clip(lower=-max_stock_w, upper=max_stock_w)

        # Apply sector caps (roughly) on absolute exposure
        w = apply_sector_caps(w, sub["sector"], sector_univ_w, sector_cap_mult)

        # Normalize gross to 1.0
        gross = float(w.abs().sum())
        if gross <= 0:
            continue
        w = w / gross

        # Final neutralization clean-up
        betas = sub["beta"].fillna(0.0)
        w = neutralize_exposures_to_beta(w, betas)

        # Renormalize gross again
        gross2 = float(w.abs().sum())
        if gross2 <= 0:
            continue
        w = w / gross2

        # Compute portfolio beta
        port_beta = float((w * betas).sum())

        # Compute forward returns
        targets = sub["target"].reindex(w.index).fillna(0.0)
        port_ret = float((w * targets).sum())
        bench_ret = float(universe["target"].mean())

        records.append(
            {
                "date": d,
                "port_ret_raw": port_ret,
                "bench_ret_raw": bench_ret,
                "beta_port_raw": port_beta,
            }
        )

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        logger.warning("Empty optimized LS backtest.")
        return bt

    # Vol targeting
    _, raw_vol, _ = annualize(bt["port_ret_raw"], rebalance_every)
    logger.info(f"Raw annualized vol (pre-scale): {raw_vol:.2%}")
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    logger.info(f"Target vol={target_vol:.2%}, scale factor={scale:.4f}")

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]
    bt["beta_port"] = bt["beta_port_raw"] * scale

    return bt


# =====================================================================
# Summary
# =====================================================================

def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float) -> None:
    if bt.empty:
        print("No results.")
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

    avg_beta = float(bt["beta_port"].mean())
    std_beta = float(bt["beta_port"].std())

    print("\n==================== OPTIMIZED LONG–SHORT (PHASE 8C) ====================")
    print(f"Target Vol (long-short): {target_vol:.2%}")
    print("-----------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Ret:    {port_ann:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("-----------------------------------------------------------------")
    print(f"BENCHMARK Total:         {bench_total:7.2%}")
    print(f"BENCHMARK Annual Ret:    {bench_ann:7.2%}")
    print(f"BENCHMARK Annual Vol:    {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max DD:        {bench_dd:7.2%}")
    print("-----------------------------------------------------------------")
    print(f"ACTIVE Annual Ret:       {active_ann:7.2%}")
    print(f"ACTIVE Annual Vol:       {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("-----------------------------------------------------------------")
    print(f"Avg portfolio beta:      {avg_beta: .4f}")
    print(f"Std portfolio beta:      {std_beta: .4f}")
    print("=================================================================\n")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 8C Optimized Long–Short Backtester.")
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

    bt = run_optimized_ls_backtest(
        con=con,
        df=df,
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
    con.close()


if __name__ == "__main__":
    main()
