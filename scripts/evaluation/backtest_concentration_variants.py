#!/usr/bin/env python3
"""
backtest_concentration_variants.py
===================================
Compare portfolio concentration strategies using the existing v2_tuned signal.

Tests signal confidence weighting: how much should we concentrate capital
in the highest-conviction picks vs spreading it evenly?

Variants tested:
    - Portfolio size: 75, 50, 30 stocks
    - Weighting power: alpha_z^1 (linear), alpha_z^1.5, alpha_z^2 (convex)
    - Position cap: 3%, 5%, 8%

Uses the same Risk4 engine with vol targeting, sector caps, and turnover control.

Usage:
    python scripts/evaluation/backtest_concentration_variants.py \
        --db data/kairos.duckdb \
        --output-dir outputs/evaluation/concentration_variants
"""

import argparse
import logging
from pathlib import Path
from itertools import product

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
# HELPERS
# =============================================================================

def safe_z(series: pd.Series) -> pd.Series:
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def max_drawdown(eq: pd.Series) -> float:
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5):
    if len(returns) == 0:
        return 0.0, 0.0, 0.0
    ann_factor = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** ann_factor - 1.0
    ann_vol = returns.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


# =============================================================================
# VARIANT DEFINITIONS
# =============================================================================

TOP_N_OPTIONS = [75, 50, 30]
POWER_OPTIONS = [1.0, 1.5, 2.0]
CAP_OPTIONS = [0.03, 0.05, 0.08]


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    top_n: int,
    weight_power: float,
    max_stock_w: float,
    rebalance_every: int = 5,
    target_vol: float = 0.25,
    adv_thresh: float = 2_000_000,
    sector_cap_mult: float = 2.0,
    lambda_tc: float = 0.5,
    turnover_cap: float = 0.30,
) -> pd.DataFrame:
    """Risk4 backtest with configurable concentration parameters."""

    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        return pd.DataFrame()

    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    records = []
    w_old = pd.Series(dtype=float)

    for d in rebal_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.copy()
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue

        picks = picks.drop_duplicates(subset="ticker", keep="first")
        picks = picks.set_index("ticker")

        # Convex weighting: alpha_z^power / vol_blend
        w_prop = picks["alpha_z"].clip(lower=0.0) ** weight_power
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / picks["vol_blend"].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / w_prop.sum()

        # Position cap
        w_prop = w_prop.clip(upper=max_stock_w)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / w_prop.sum()

        # Sector caps
        sector_counts = universe.groupby("sector")["ticker"].count()
        total_univ = float(len(universe))
        sector_univ_w = sector_counts / total_univ

        for sec in picks["sector"].unique():
            mask = picks.index[picks["sector"] == sec]
            sec_w = w_prop.reindex(mask).sum()
            cap = sector_univ_w.get(sec, 0.0) * sector_cap_mult
            if sec_w > cap and cap > 0:
                w_prop[mask] *= cap / sec_w

        if w_prop.sum() <= 0:
            continue
        w_prop = w_prop / w_prop.sum()

        # Turnover control
        w_prop = w_prop[~w_prop.index.duplicated(keep="first")]
        if w_old.empty:
            w_new = w_prop.copy()
        else:
            all_tickers = pd.Index(w_old.index).union(w_prop.index).unique()
            w_old_full = w_old.reindex(all_tickers).fillna(0.0)
            w_prop_full = w_prop.reindex(all_tickers).fillna(0.0)
            w_pre = (1 - lambda_tc) * w_old_full + lambda_tc * w_prop_full
            turnover = float((w_pre - w_old_full).abs().sum())
            if turnover > turnover_cap:
                scale = turnover_cap / turnover
                w_new_full = w_old_full + scale * (w_pre - w_old_full)
            else:
                w_new_full = w_pre
            if w_new_full.sum() <= 0:
                w_new_full = w_old_full
            w_new_full = w_new_full / w_new_full.sum()
            w_new = w_new_full.reindex(w_prop.index).fillna(0.0)

        w_new = w_new[~w_new.index.duplicated(keep="first")]
        w_old = w_new.copy()

        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue

        port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        bench_ret = float(universe["target"].mean())

        # Track concentration
        top5_w = float(w_new.nlargest(5).sum())
        top10_w = float(w_new.nlargest(10).sum())

        records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
            "top5_weight": top5_w,
            "top10_weight": top10_w,
        })

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt

    # Vol targeting
    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0

    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare portfolio concentration strategies"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", default="alpha_ml_v2_tuned_clf")
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-11-01")
    parser.add_argument("--target-vol", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CONCENTRATION VARIANT COMPARISON")
    logger.info("  Signal confidence weighting analysis")
    logger.info("=" * 70)

    con = duckdb.connect(args.db, read_only=True)

    logger.info("Loading data...")
    df = con.execute(f"""
        SELECT
            fm.ticker, fm.date,
            fm.{args.alpha_column} AS alpha,
            fm.{args.target_column} AS target,
            fm.vol_blend, fm.adv_20,
            t.sector
        FROM feat_matrix_v2 fm
        LEFT JOIN (
            SELECT DISTINCT ticker, sector
            FROM tickers
            WHERE sector IS NOT NULL AND ticker != 'N/A'
        ) t USING (ticker)
        WHERE fm.{args.alpha_column} IS NOT NULL
          AND fm.{args.target_column} IS NOT NULL
          AND fm.vol_blend IS NOT NULL
          AND fm.adv_20 IS NOT NULL
          AND fm.date >= '{args.start_date}'
          AND fm.date <= '{args.end_date}'
        ORDER BY fm.date, fm.ticker
    """).fetchdf()
    con.close()
    logger.info(f"  Loaded {len(df):,} rows")

    # Run all variants
    results = []
    variants = list(product(TOP_N_OPTIONS, POWER_OPTIONS, CAP_OPTIONS))
    logger.info(f"  Running {len(variants)} variants...")

    for i, (top_n, power, cap) in enumerate(variants):
        label = f"n{top_n}_p{power:.1f}_c{int(cap*100)}pct"
        logger.info(f"  [{i+1}/{len(variants)}] {label}")

        bt = run_backtest(
            df.copy(),
            top_n=top_n,
            weight_power=power,
            max_stock_w=cap,
            target_vol=args.target_vol,
        )

        if bt.empty:
            continue

        ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], 5)
        eq = (1 + bt["port_ret"]).cumprod()
        mdd = max_drawdown(eq)
        total_ret = float(eq.iloc[-1] - 1)
        avg_top5 = bt["top5_weight"].mean()
        avg_top10 = bt["top10_weight"].mean()

        results.append({
            "variant": label,
            "top_n": top_n,
            "weight_power": power,
            "position_cap": cap,
            "sharpe": sharpe,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "max_drawdown": mdd,
            "total_return": total_ret,
            "avg_top5_weight": avg_top5,
            "avg_top10_weight": avg_top10,
        })

    results_df = pd.DataFrame(results).sort_values("sharpe", ascending=False)

    # Print comparison
    print(f"\n\n{'=' * 110}")
    print("  CONCENTRATION VARIANT COMPARISON (sorted by Sharpe)")
    print(f"{'=' * 110}")
    print(f"{'Variant':<22} {'N':>3} {'Pwr':>4} {'Cap':>5} {'Sharpe':>7} "
          f"{'AnnRet':>8} {'MaxDD':>7} {'TotRet':>9} {'Top5%':>6} {'Top10%':>7}")
    print("-" * 110)

    for _, r in results_df.iterrows():
        print(f"{r['variant']:<22} {int(r['top_n']):>3} {r['weight_power']:>4.1f} "
              f"{r['position_cap']:>5.0%} {r['sharpe']:>7.3f} "
              f"{r['ann_return']:>8.1%} {r['max_drawdown']:>7.1%} "
              f"{r['total_return']:>9.1%} {r['avg_top5_weight']:>6.1%} "
              f"{r['avg_top10_weight']:>7.1%}")

    # Best variant
    best = results_df.iloc[0]
    baseline = results_df[
        (results_df['top_n'] == 75) &
        (results_df['weight_power'] == 1.0) &
        (results_df['position_cap'] == 0.03)
    ]

    print(f"\n  BEST: {best['variant']}  Sharpe={best['sharpe']:.3f}")
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f"  BASE: {b['variant']}  Sharpe={b['sharpe']:.3f}")
        print(f"  Delta: {best['sharpe'] - b['sharpe']:+.3f} Sharpe")

    # Save
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "concentration_variant_comparison.csv", index=False)
        logger.info(f"\nSaved to: {out / 'concentration_variant_comparison.csv'}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
