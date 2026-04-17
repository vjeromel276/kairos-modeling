#!/usr/bin/env python3
"""
prediction_ev_analysis.py
=========================
Prediction EV (Expected Value) evaluation module.

This is the missing bridge between model output and portfolio outcome,
as defined in Phase 3 of KAIROS_OBJECTIVE_AND_PROMOTION_POLICY_v2.md.

For each prediction bucket (decile), this module computes:
    - Count, win rate
    - Average return, median return
    - Average win, average loss
    - Expected value per bucket
    - Cumulative return and max drawdown by bucket
    - All metrics split by regime (vol regime, trend regime)

Answers three key questions:
    1. "Where does the money come from?"
    2. "Which confidence bands are actually worth trading?"
    3. "Does the model keep positive EV across regimes?"

Usage:
    python scripts/evaluation/prediction_ev_analysis.py \
        --db data/kairos.duckdb \
        --alpha-column alpha_ml_v3_neutral_clf \
        --start-date 2015-01-01 \
        --end-date 2025-12-12

    # Use a specific alpha table instead of feat_matrix_v2:
    python scripts/evaluation/prediction_ev_analysis.py \
        --db data/kairos.duckdb \
        --alpha-column alpha_ml_v3_neutral_clf \
        --alpha-table feat_alpha_ml_xgb_v3_neutral

    # Output CSV reports:
    python scripts/evaluation/prediction_ev_analysis.py \
        --db data/kairos.duckdb \
        --output-dir outputs/evaluation/v3_neutral
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CORE METRICS
# =============================================================================

def compute_bucket_metrics(df: pd.DataFrame, bucket_col: str = 'bucket') -> pd.DataFrame:
    """
    Compute EV metrics for each prediction bucket.

    Returns one row per bucket with: count, win_rate, avg_return, median_return,
    avg_win, avg_loss, expected_value, cumulative_return, max_drawdown, sharpe.
    """
    results = []

    for bucket, group in df.groupby(bucket_col):
        returns = group['fwd_ret'].values
        n = len(returns)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        win_rate = len(wins) / n if n > 0 else 0
        avg_return = returns.mean()
        median_return = np.median(returns)
        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        ev = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Cumulative return and drawdown (daily bucket-average returns)
        daily_rets = group.groupby('date')['fwd_ret'].mean().sort_index()
        cum = (1 + daily_rets).cumprod()
        cumulative_return = cum.iloc[-1] - 1 if len(cum) > 0 else 0
        running_max = cum.cummax()
        drawdown = (cum / running_max - 1)
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Sharpe (annualized from 5-day returns)
        ret_std = daily_rets.std()
        sharpe = (daily_rets.mean() / ret_std * np.sqrt(52)) if ret_std > 0 else 0

        results.append({
            'bucket': bucket,
            'count': n,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'median_return': median_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'expected_value': ev,
            'cumulative_return': cumulative_return,
            'max_drawdown': max_drawdown,
            'sharpe': sharpe,
        })

    return pd.DataFrame(results)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(
    con: duckdb.DuckDBPyConnection,
    alpha_column: str,
    alpha_table: str,
    start_date: str,
    end_date: str,
    adv_thresh: float,
    n_buckets: int,
) -> pd.DataFrame:
    """
    Load predictions, forward returns, and regime labels.
    Rank predictions into buckets per date.
    """

    logger.info(f"Loading data...")
    logger.info(f"  Alpha: {alpha_column} from {alpha_table}")
    logger.info(f"  Period: {start_date} to {end_date}")
    logger.info(f"  ADV threshold: ${adv_thresh:,.0f}")

    # Build query depending on whether alpha is in feat_matrix_v2 or separate table
    if alpha_table == 'feat_matrix_v2':
        query = f"""
            SELECT
                m.ticker,
                m.date,
                m.{alpha_column} as alpha,
                m.ret_5d_f as fwd_ret,
                m.adv_20,
                r.vol_regime,
                r.trend_regime,
                r.regime,
                NTILE({n_buckets}) OVER (
                    PARTITION BY m.date ORDER BY m.{alpha_column}
                ) as bucket
            FROM feat_matrix_v2 m
            LEFT JOIN regime_history_academic r ON m.date = r.date
            WHERE m.{alpha_column} IS NOT NULL
              AND m.ret_5d_f IS NOT NULL
              AND m.adv_20 >= {adv_thresh}
              AND m.date >= '{start_date}'
              AND m.date <= '{end_date}'
        """
    else:
        query = f"""
            SELECT
                a.ticker,
                a.date,
                a.{alpha_column} as alpha,
                t.ret_5d_f as fwd_ret,
                m.adv_20,
                r.vol_regime,
                r.trend_regime,
                r.regime,
                NTILE({n_buckets}) OVER (
                    PARTITION BY a.date ORDER BY a.{alpha_column}
                ) as bucket
            FROM {alpha_table} a
            JOIN feat_targets t ON a.ticker = t.ticker AND a.date = t.date
            LEFT JOIN feat_matrix_v2 m ON a.ticker = m.ticker AND a.date = m.date
            LEFT JOIN regime_history_academic r ON a.date = r.date
            WHERE a.{alpha_column} IS NOT NULL
              AND t.ret_5d_f IS NOT NULL
              AND (m.adv_20 IS NULL OR m.adv_20 >= {adv_thresh})
              AND a.date >= '{start_date}'
              AND a.date <= '{end_date}'
        """

    df = con.execute(query).fetchdf()
    df['date'] = pd.to_datetime(df['date'])

    logger.info(f"  Loaded {len(df):,} rows")
    logger.info(f"  Dates: {df['date'].min().date()} to {df['date'].max().date()}")
    logger.info(f"  Tickers: {df['ticker'].nunique():,}")
    logger.info(f"  Buckets: {df['bucket'].nunique()}")

    return df


# =============================================================================
# REPORTING
# =============================================================================

def print_bucket_report(metrics: pd.DataFrame, title: str):
    """Print formatted bucket metrics."""

    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")

    fmt = (
        "{bucket:>7} {count:>9,} {win_rate:>9.1%} {avg_return:>11.4f} "
        "{median_return:>11.4f} {avg_win:>10.4f} {avg_loss:>10.4f} "
        "{expected_value:>9.4f} {sharpe:>8.2f} {max_drawdown:>8.1%}"
    )
    header = (
        f"{'Bucket':>7} {'Count':>9} {'Win Rate':>9} {'Avg Ret':>11} "
        f"{'Med Ret':>11} {'Avg Win':>10} {'Avg Loss':>10} "
        f"{'EV':>9} {'Sharpe':>8} {'Max DD':>8}"
    )
    print(header)
    print("-" * 90)

    for _, row in metrics.iterrows():
        print(fmt.format(**row.to_dict()))

    # Summary
    top = metrics[metrics['bucket'] == metrics['bucket'].max()].iloc[0]
    bottom = metrics[metrics['bucket'] == metrics['bucket'].min()].iloc[0]
    print("-" * 90)
    print(f"  Top bucket EV: {top['expected_value']:.4f}  |  "
          f"Bottom bucket EV: {bottom['expected_value']:.4f}  |  "
          f"Spread: {top['avg_return'] - bottom['avg_return']:.4f}")


def run_regime_analysis(df: pd.DataFrame, regime_col: str, regime_label: str):
    """Run bucket analysis split by a regime dimension."""

    print(f"\n\n{'#' * 90}")
    print(f"  REGIME SPLIT: {regime_label}")
    print(f"{'#' * 90}")

    regimes = sorted(df[regime_col].dropna().unique())
    regime_summaries = []

    for regime in regimes:
        subset = df[df[regime_col] == regime]
        if len(subset) < 100:
            print(f"\n  {regime}: too few observations ({len(subset)}), skipping")
            continue

        metrics = compute_bucket_metrics(subset)
        print_bucket_report(metrics, f"{regime_label}: {regime} (n={len(subset):,})")

        top = metrics[metrics['bucket'] == metrics['bucket'].max()].iloc[0]
        regime_summaries.append({
            'regime': regime,
            'n_obs': len(subset),
            'top_bucket_ev': top['expected_value'],
            'top_bucket_sharpe': top['sharpe'],
            'top_bucket_win_rate': top['win_rate'],
            'top_bucket_max_dd': top['max_drawdown'],
        })

    if regime_summaries:
        summary_df = pd.DataFrame(regime_summaries)
        print(f"\n\n  {regime_label} SUMMARY (top bucket across regimes)")
        print(f"  {'-' * 70}")
        for _, row in summary_df.iterrows():
            print(f"  {row['regime']:<25} EV={row['top_bucket_ev']:+.4f}  "
                  f"Sharpe={row['top_bucket_sharpe']:.2f}  "
                  f"WinRate={row['top_bucket_win_rate']:.1%}  "
                  f"MaxDD={row['top_bucket_max_dd']:.1%}")

    return regime_summaries


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prediction EV analysis — bucket predictions and compute "
                    "expected value metrics split by regime"
    )
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--alpha-column", default="alpha_ml_v3_neutral_clf",
                        help="Column name for alpha signal")
    parser.add_argument("--alpha-table", default="feat_matrix_v2",
                        help="Table containing alpha signal (default: feat_matrix_v2)")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-12-12",
                        help="End date (research DB clean boundary)")
    parser.add_argument("--adv-thresh", type=float, default=2_000_000,
                        help="Minimum average daily volume")
    parser.add_argument("--n-buckets", type=int, default=10,
                        help="Number of prediction buckets (default: 10 deciles)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save CSV reports (optional)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("PREDICTION EV ANALYSIS")
    logger.info("  Phase 3 of Kairos Objective & Promotion Policy")
    logger.info("=" * 70)

    con = duckdb.connect(args.db, read_only=True)

    try:
        # Load data with bucket assignments
        df = load_data(
            con, args.alpha_column, args.alpha_table,
            args.start_date, args.end_date,
            args.adv_thresh, args.n_buckets,
        )

        if len(df) == 0:
            logger.error("No data loaded. Check alpha column and date range.")
            return

        # --- Overall bucket analysis ---
        overall_metrics = compute_bucket_metrics(df)
        print_bucket_report(overall_metrics, "OVERALL BUCKET ANALYSIS")

        # --- Regime splits ---
        vol_summaries = run_regime_analysis(df, 'vol_regime', 'Volatility Regime')
        trend_summaries = run_regime_analysis(df, 'trend_regime', 'Trend Regime')
        full_summaries = run_regime_analysis(df, 'regime', 'Full Regime (Vol x Trend)')

        # --- Final verdict ---
        print(f"\n\n{'=' * 90}")
        print("  SIGNAL TRIAGE VERDICT")
        print(f"{'=' * 90}")

        top_bucket = overall_metrics[
            overall_metrics['bucket'] == overall_metrics['bucket'].max()
        ].iloc[0]
        bottom_bucket = overall_metrics[
            overall_metrics['bucket'] == overall_metrics['bucket'].min()
        ].iloc[0]

        print(f"\n  Top bucket (long signal):")
        print(f"    EV:       {top_bucket['expected_value']:+.4f}")
        print(f"    Win Rate: {top_bucket['win_rate']:.1%}")
        print(f"    Sharpe:   {top_bucket['sharpe']:.2f}")
        print(f"    Max DD:   {top_bucket['max_drawdown']:.1%}")

        print(f"\n  Bottom bucket (short signal):")
        print(f"    EV:       {bottom_bucket['expected_value']:+.4f}")
        print(f"    Win Rate: {bottom_bucket['win_rate']:.1%}")
        print(f"    Sharpe:   {bottom_bucket['sharpe']:.2f}")
        print(f"    Max DD:   {bottom_bucket['max_drawdown']:.1%}")

        # Regime robustness check
        if vol_summaries:
            all_positive_ev = all(s['top_bucket_ev'] > 0 for s in vol_summaries)
            print(f"\n  Top bucket positive EV in all vol regimes: "
                  f"{'YES' if all_positive_ev else 'NO'}")
            if not all_positive_ev:
                bad = [s['regime'] for s in vol_summaries if s['top_bucket_ev'] <= 0]
                print(f"    WARNING: Negative EV in: {', '.join(bad)}")

        if trend_summaries:
            all_positive_ev = all(s['top_bucket_ev'] > 0 for s in trend_summaries)
            print(f"  Top bucket positive EV in all trend regimes: "
                  f"{'YES' if all_positive_ev else 'NO'}")
            if not all_positive_ev:
                bad = [s['regime'] for s in trend_summaries if s['top_bucket_ev'] <= 0]
                print(f"    WARNING: Negative EV in: {', '.join(bad)}")

        # --- Save CSVs ---
        if args.output_dir:
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)

            overall_metrics.to_csv(out / 'bucket_metrics_overall.csv', index=False)
            logger.info(f"  Saved: {out / 'bucket_metrics_overall.csv'}")

            # Per-regime bucket metrics
            for regime_col, label in [
                ('vol_regime', 'vol'),
                ('trend_regime', 'trend'),
                ('regime', 'full'),
            ]:
                regime_rows = []
                for regime in sorted(df[regime_col].dropna().unique()):
                    subset = df[df[regime_col] == regime]
                    if len(subset) < 100:
                        continue
                    m = compute_bucket_metrics(subset)
                    m['regime'] = regime
                    regime_rows.append(m)
                if regime_rows:
                    regime_df = pd.concat(regime_rows, ignore_index=True)
                    fname = out / f'bucket_metrics_by_{label}_regime.csv'
                    regime_df.to_csv(fname, index=False)
                    logger.info(f"  Saved: {fname}")

            logger.info(f"\n  All reports saved to: {out}")

    finally:
        con.close()

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
