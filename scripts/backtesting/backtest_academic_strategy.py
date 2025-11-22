#!/usr/bin/env python3
"""
backtest_academic_strategy.py

Clean academic backtester for factor signals using:
    - feat_matrix       (features + targets)
    - regime_history_academic  (optional regime filter)

It:
    - Ranks stocks by an alpha column on each rebalance date
    - Builds an equal-weight long or long/short portfolio
    - Uses a forward-return target column (default: ret_5d_f)
    - Computes portfolio vs. cross-sectional benchmark
    - Reports total return, annualized return, volatility, Sharpe, max drawdown

Usage example:

    python scripts/backtest_academic_strategy.py \
        --db data/kairos.duckdb \
        --alpha-column price_vs_sma_21 \
        --target-column ret_5d_f \
        --top-n 50 \
        --rebalance-every 5 \
        --start-date 2015-01-01 \
        --end-date 2025-01-01 \
        --regime-filter normal_vol_neutral,low_vol_neutral

"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np
from typing import Optional, List


# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------
def compute_max_drawdown(equity: pd.Series) -> float:
    """
    Equity is a series of cumulative returns in gross form (starting at 1.0).
    Returns max drawdown as a negative float, e.g. -0.35 = -35%.
    """
    running_max = equity.cummax()
    drawdowns = (equity / running_max) - 1.0
    return float(drawdowns.min())


def annualize_stats(returns: pd.Series, period_len_days: int = 5) -> tuple[float, float, float]:
    """
    Annualize return, volatility, and Sharpe from per-period returns.
    Assumes each period_len_days is ~trading days (default 5 for weekly rebalance).
    """
    if returns.empty:
        return 0.0, 0.0, 0.0

    # Total return
    gross = (1.0 + returns).prod()
    n_periods = len(returns)
    total_days = n_periods * period_len_days
    if total_days == 0:
        return 0.0, 0.0, 0.0

    annual_factor = 252.0 / period_len_days
    avg_period_ret = gross ** (1.0 / n_periods) - 1.0
    annual_ret = (1.0 + avg_period_ret) ** annual_factor - 1.0

    # Vol
    period_vol = returns.std()
    annual_vol = period_vol * np.sqrt(annual_factor)

    sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
    return float(annual_ret), float(annual_vol), float(sharpe)


# ---------------------------------------------------------------------
# Core backtest logic
# ---------------------------------------------------------------------
def load_backtest_data(
    con: duckdb.DuckDBPyConnection,
    alpha_column: str,
    target_column: str,
    start_date: Optional[str],
    end_date: Optional[str],
    regimes: Optional[List[str]]
) -> pd.DataFrame:
    """
    Loads feat_matrix and joins regime_history_academic on date.
    Filters by date and optional regimes.
    """

    # Verify columns exist in feat_matrix
    cols_df = con.execute("PRAGMA table_info('feat_matrix')").fetchdf()
    cols = cols_df["name"].tolist()
    if alpha_column not in cols:
        raise RuntimeError(f"Alpha column '{alpha_column}' not found in feat_matrix")
    if target_column not in cols:
        raise RuntimeError(f"Target column '{target_column}' not found in feat_matrix")

    # Build WHERE clauses
    where_clauses = [f"fm.{alpha_column} IS NOT NULL", f"fm.{target_column} IS NOT NULL"]

    if start_date:
        where_clauses.append(f"fm.date >= DATE '{start_date}'")
    if end_date:
        where_clauses.append(f"fm.date <= DATE '{end_date}'")

    where_sql = " AND ".join(where_clauses)

    # Join feat_matrix with regime history
    logger.info("Loading feat_matrix + regime_history_academic...")
    base_query = f"""
        SELECT
            fm.ticker,
            fm.date,
            fm.{alpha_column} AS alpha,
            fm.{target_column} AS target,
            rh.regime
        FROM feat_matrix fm
        LEFT JOIN regime_history_academic rh
          ON fm.date = rh.date
        WHERE {where_sql}
    """

    df = con.execute(base_query).fetchdf()
    if df.empty:
        raise RuntimeError("No data returned for given filters. Check dates, columns, and regimes.")

    # Apply regime filter in pandas if requested
    if regimes:
        before = len(df)
        df = df[df["regime"].isin(regimes)]
        logger.info(f"Applied regime filter {regimes}: {before} → {len(df)} rows")

    df = df.dropna(subset=["alpha", "target"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(f"Backtest dataset: {len(df):,} rows, {df['date'].nunique()} dates.")
    return df


def run_backtest(
    df: pd.DataFrame,
    top_n: int,
    bottom_n: int,
    rebalance_every: int,
    alpha_higher_is_better: bool
) -> pd.DataFrame:
    """
    Runs a simple cross-sectional rank-based backtest on the provided dataframe
    with columns: date, ticker, alpha, target, regime.
    """

    logger.info("Running backtest...")
    all_dates = sorted(df["date"].unique())
    if len(all_dates) < 10:
        logger.warning("Very few dates available for backtest.")

    rebalance_dates = all_dates[::rebalance_every]

    records = []

    for d in rebalance_dates:
        slice_df = df[df["date"] == d]
        if slice_df.empty:
            continue

        tmp = slice_df.copy()

        # Rank: higher alpha is better or worse?
        tmp = tmp.sort_values("alpha", ascending=not alpha_higher_is_better)

        long_df = tmp.head(top_n) if top_n > 0 else tmp.iloc[0:0]
        short_df = tmp.tail(bottom_n) if bottom_n > 0 else tmp.iloc[0:0]

        # Portfolio returns
        long_ret = long_df["target"].mean() if not long_df.empty else 0.0
        short_ret = short_df["target"].mean() if not short_df.empty else 0.0

        if bottom_n > 0:
            port_ret = long_ret - short_ret
        else:
            port_ret = long_ret

        # Benchmark = average target across all stocks that day
        bench_ret = tmp["target"].mean()

        records.append(
            {
                "date": d,
                "n_universe": len(tmp),
                "n_long": len(long_df),
                "n_short": len(short_df),
                "port_ret": port_ret,
                "bench_ret": bench_ret,
                "active_ret": port_ret - bench_ret,
            }
        )

    bt_df = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    logger.info(f"Backtest produced {len(bt_df)} rebalance periods.")
    return bt_df


def summarize_backtest(bt_df: pd.DataFrame, rebalance_every: int):
    """
    Print summary stats and basic performance metrics.
    """

    if bt_df.empty:
        logger.warning("Backtest dataframe is empty; nothing to summarize.")
        return

    port_rets = bt_df["port_ret"]
    bench_rets = bt_df["bench_ret"]
    active_rets = bt_df["active_ret"]

    # Equity curves (gross)
    port_equity = (1.0 + port_rets).cumprod()
    bench_equity = (1.0 + bench_rets).cumprod()
    active_equity = (1.0 + active_rets).cumprod()

    port_total = float(port_equity.iloc[-1] - 1.0)
    bench_total = float(bench_equity.iloc[-1] - 1.0)
    active_total = float(active_equity.iloc[-1] - 1.0)

    port_ann, port_vol, port_sharpe = annualize_stats(port_rets, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize_stats(bench_rets, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize_stats(active_rets, rebalance_every)

    port_dd = compute_max_drawdown(port_equity)
    bench_dd = compute_max_drawdown(bench_equity)

    print("\n================ ACADEMIC BACKTEST SUMMARY ================")
    print(f"Periods:                 {len(bt_df)}")
    print(f"Rebalance freq (days):   {rebalance_every}")
    print("-----------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Ann. Return:   {port_ann:7.2%}")
    print(f"PORTFOLIO Ann. Vol:      {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max Drawdown:  {port_dd:7.2%}")
    print("-----------------------------------------------------------")
    print(f"BENCHMARK Total Return:  {bench_total:7.2%}")
    print(f"BENCHMARK Ann. Return:   {bench_ann:7.2%}")
    print(f"BENCHMARK Ann. Vol:      {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max Drawdown:  {bench_dd:7.2%}")
    print("-----------------------------------------------------------")
    print(f"ACTIVE Ann. Return:      {active_ann:7.2%}")
    print(f"ACTIVE Ann. Vol:         {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("===========================================================\n")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Clean academic factor backtester")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--alpha-column", required=True, help="Column name in feat_matrix to use as alpha")
    parser.add_argument("--target-column", default="ret_5d_f", help="Forward return column (feat_targets) to use")
    parser.add_argument("--top-n", type=int, default=50, help="Number of long positions")
    parser.add_argument("--bottom-n", type=int, default=0, help="Number of short positions (0 = long-only)")
    parser.add_argument("--rebalance-every", type=int, default=5, help="Rebalance frequency in days (trading)")
    parser.add_argument("--alpha-higher-is-better", action="store_true", help="If set, higher alpha rank is better")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--regime-filter",
        type=str,
        default="",
        help="Comma-separated list of regimes from regime_history_academic.regime to include (optional)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    con = duckdb.connect(args.db)

    regimes = [r.strip() for r in args.regime_filter.split(",") if r.strip()] if args.regime_filter else None

    df = load_backtest_data(
        con=con,
        alpha_column=args.alpha_column,
        target_column=args.target_column,
        start_date=args.start_date,
        end_date=args.end_date,
        regimes=regimes
    )

    bt_df = run_backtest(
        df=df,
        top_n=args.top_n,
        bottom_n=args.bottom_n,
        rebalance_every=args.rebalance_every,
        alpha_higher_is_better=args.alpha_higher_is_better
    )

    summarize_backtest(bt_df, args.rebalance_every)

    # Optionally, you can save bt_df back to DuckDB for later analysis
    con.execute("DROP TABLE IF EXISTS backtest_results_academic")
    con.register("bt_df", bt_df)
    con.execute("CREATE TABLE backtest_results_academic AS SELECT * FROM bt_df")
    logger.info("Saved backtest_results_academic to DuckDB.")

    con.close()


if __name__ == "__main__":
    main()
