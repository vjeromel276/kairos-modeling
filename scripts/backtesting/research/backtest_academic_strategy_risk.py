#!/usr/bin/env python3
"""
backtest_academic_strategy_risk.py

Risk-Controlled Backtester v1

Adds:
    - Cross-sectional z-scoring of alpha per date
    - Clipping (winsorization) of alpha z-scores at ±3
    - Top-N selection by clipped z-score
    - Equal-weighted long-only portfolio
    - Volatility targeting to a desired annual volatility (e.g. 20%)

Assumptions:
    - feat_matrix contains:
        * ticker
        * date
        * alpha column (e.g., alpha_composite_v3)
        * target column (e.g., ret_5d_f)
    - Each row is one (ticker, date) snapshot.
    - target_column is a forward 5-day return (consistent with rebalance_every=5).

Usage example:

    python scripts/backtesting/backtest_academic_strategy_risk.py \
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
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def safe_zscore(series: pd.Series) -> pd.Series:
    """Compute z-score safely for a cross-section on one date."""
    mean = series.mean()
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mean) / std


def compute_max_drawdown(equity: pd.Series) -> float:
    """Compute max drawdown on equity curve."""
    running_max = equity.cummax()
    dd = (equity / running_max) - 1.0
    return float(dd.min())


def annualize_stats(returns: pd.Series, period_len_days: int = 5) -> tuple[float, float, float]:
    """Annualized return, vol, Sharpe from periodic returns."""
    if returns.empty:
        return 0.0, 0.0, 0.0

    gross = (1.0 + returns).prod()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0, 0.0, 0.0

    # Annualization factor for 5-day steps
    annual_factor = 252.0 / period_len_days

    avg_period_ret = gross ** (1.0 / n_periods) - 1.0
    annual_ret = (1.0 + avg_period_ret) ** annual_factor - 1.0

    period_vol = returns.std()
    annual_vol = period_vol * np.sqrt(annual_factor)

    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
    return float(annual_ret), float(annual_vol), float(sharpe)


def load_data(
    con: duckdb.DuckDBPyConnection,
    alpha_column: str,
    target_column: str,
    start_date: str | None,
    end_date: str | None
) -> pd.DataFrame:
    """Load alpha + target + date/ticker from feat_matrix."""
    logger.info("Loading feat_matrix for risk-controlled backtest...")

    where_clauses = [
        f"{alpha_column} IS NOT NULL",
        f"{target_column} IS NOT NULL",
    ]
    if start_date:
        where_clauses.append(f"date >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"date <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)

    df = con.execute(f"""
        SELECT
            ticker,
            date,
            {alpha_column} AS alpha,
            {target_column} AS target
        FROM feat_matrix
        WHERE {where_sql}
        ORDER BY date, ticker
    """).fetchdf()

    logger.info(f"Loaded {len(df):,} rows, {df['date'].nunique()} dates.")
    return df


def run_risk_controlled_backtest(
    df: pd.DataFrame,
    top_n: int,
    rebalance_every: int,
    target_vol: float
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Per-date:
        - zscore alpha
        - clip at ±3
        - pick top N
        - equal weight portfolio
    Then:
        - compute 5-day returns
        - scale to target_vol
    """

    logger.info("Applying cross-sectional z-scoring and clipping...")

    # Cross-sectional z-score per date
    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_zscore)
    # Clip outliers
    df["alpha_z_clipped"] = df["alpha_z"].clip(-3.0, 3.0)

    # Build portfolio returns at rebalance dates
    all_dates = sorted(df["date"].unique())
    rebalance_dates = all_dates[::rebalance_every]

    records = []
    for d in rebalance_dates:
        slice_df = df[df["date"] == d].copy()
        if slice_df.empty:
            continue

        # Rank by clipped z
        slice_df = slice_df.sort_values("alpha_z_clipped", ascending=False)
        picks = slice_df.head(top_n)

        if picks.empty:
            continue

        # Equal weights long-only
        w = 1.0 / len(picks)

        port_ret = float((picks["target"] * w).sum())
        bench_ret = float(slice_df["target"].mean())

        records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
        })

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        logger.warning("No backtest periods constructed.")
        return bt, pd.Series(dtype=float)

    # Vol targeting
    logger.info("Computing vol scaling factor for target volatility...")

    raw_returns = bt["port_ret_raw"]
    # Realized vol of raw returns (5-day returns)
    _, raw_ann_vol, _ = annualize_stats(raw_returns, period_len_days=rebalance_every)
    logger.info(f"Raw annualized vol: {raw_ann_vol:.2%}")

    if raw_ann_vol > 0:
        scale = target_vol / raw_ann_vol
    else:
        scale = 1.0

    logger.info(f"Vol target: {target_vol:.2%}, scale factor: {scale:.4f}")

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]  # benchmark left unscaled
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt, raw_returns


def summarize(bt: pd.DataFrame, rebalance_every: int, target_vol: float) -> None:
    """Print summary stats for risk-controlled backtest."""

    if bt.empty:
        logger.warning("Empty backtest results; nothing to summarize.")
        return

    port = bt["port_ret"]
    bench = bt["bench_ret"]
    active = bt["active_ret"]

    port_equity = (1.0 + port).cumprod()
    bench_equity = (1.0 + bench).cumprod()

    port_total = float(port_equity.iloc[-1] - 1.0)
    bench_total = float(bench_equity.iloc[-1] - 1.0)

    port_ann, port_vol, port_sharpe = annualize_stats(port, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize_stats(bench, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize_stats(active, rebalance_every)

    port_dd = compute_max_drawdown(port_equity)
    bench_dd = compute_max_drawdown(bench_equity)

    print("\n================ RISK-CONTROLLED BACKTEST SUMMARY ================")
    print(f"Periods:                 {len(bt)}")
    print(f"Rebalance freq (days):   {rebalance_every}")
    print(f"Target Vol:              {target_vol:.2%}")
    print("-----------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Ann. Return:   {port_ann:7.2%}")
    print(f"PORTFOLIO Ann. Vol:      {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max Drawdown:  {port_dd:7.2%}")
    print("-----------------------------------------------------------------")
    print(f"BENCHMARK Total Return:  {bench_total:7.2%}")
    print(f"BENCHMARK Ann. Return:   {bench_ann:7.2%}")
    print(f"BENCHMARK Ann. Vol:      {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max Drawdown:  {bench_dd:7.2%}")
    print("-----------------------------------------------------------------")
    print(f"ACTIVE Ann. Return:      {active_ann:7.2%}")
    print(f"ACTIVE Ann. Vol:         {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("=================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Risk-controlled academic backtest (v1).")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--alpha-column", required=True, help="Alpha column in feat_matrix (e.g. alpha_composite_v3)")
    parser.add_argument("--target-column", default="ret_5d_f", help="Target column in feat_matrix (default ret_5d_f)")
    parser.add_argument("--top-n", type=int, default=75)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--target-vol", type=float, default=0.20, help="Target annualized volatility (e.g. 0.20)")
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    df = load_data(
        con,
        alpha_column=args.alpha_column,
        target_column=args.target_column,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    bt, raw_returns = run_risk_controlled_backtest(
        df,
        top_n=args.top_n,
        rebalance_every=args.rebalance_every,
        target_vol=args.target_vol,
    )

    summarize(bt, rebalance_every=args.rebalance_every, target_vol=args.target_vol)

    con.close()


if __name__ == "__main__":
    main()
