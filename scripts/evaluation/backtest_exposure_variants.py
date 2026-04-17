#!/usr/bin/env python3
"""
backtest_exposure_variants.py
=============================
Compare portfolio performance under different regime-aware exposure rules.

This is Step 6 of the Kairos Objective & Promotion Policy roadmap:
    "Backtest exposure rules with existing portfolio framework and compare:
     current long-only, current regime switch, long-only + vol gating,
     long-only + drawdown throttle"

Uses the existing Risk4 backtest engine with regime-conditioned exposure scaling.

Usage:
    python scripts/evaluation/backtest_exposure_variants.py \
        --db data/kairos.duckdb \
        --alpha-column alpha_ml_v2_tuned_clf \
        --start-date 2015-01-01 \
        --end-date 2025-11-01

    # Save comparison CSV:
    python scripts/evaluation/backtest_exposure_variants.py \
        --db data/kairos.duckdb \
        --output-dir outputs/evaluation/exposure_variants
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
# HELPERS (from Risk4 backtester)
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
# EXPOSURE VARIANT DEFINITIONS
# =============================================================================

VARIANTS = {
    "baseline": {
        "description": "Full exposure, no regime gating",
        "exposure_fn": lambda vol_regime: 1.0,
    },
    "vol_gate_70": {
        "description": "Low-vol regime -> 70% exposure",
        "exposure_fn": lambda vol_regime: 0.70 if vol_regime == "low_vol" else 1.0,
    },
    "vol_gate_50": {
        "description": "Low-vol regime -> 50% exposure",
        "exposure_fn": lambda vol_regime: 0.50 if vol_regime == "low_vol" else 1.0,
    },
    "vol_gate_zero": {
        "description": "Low-vol regime -> 0% (cash), high/normal -> 100%",
        "exposure_fn": lambda vol_regime: 0.0 if vol_regime == "low_vol" else 1.0,
    },
    "high_vol_only": {
        "description": "Only invest in high-vol regime",
        "exposure_fn": lambda vol_regime: 1.0 if vol_regime == "high_vol" else 0.0,
    },
    "high_normal_only": {
        "description": "High + normal vol -> 100%, low vol -> 0%",
        "exposure_fn": lambda vol_regime: 0.0 if vol_regime == "low_vol" else 1.0,
    },
}


# =============================================================================
# CORE BACKTEST (Risk4 engine with exposure scaling)
# =============================================================================

def run_backtest_with_exposure(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    exposure_fn,
    top_n: int = 75,
    rebalance_every: int = 5,
    target_vol: float = 0.25,
    adv_thresh: float = 2_000_000,
    max_stock_w: float = 0.03,
    sector_cap_mult: float = 2.0,
    lambda_tc: float = 0.5,
    turnover_cap: float = 0.30,
) -> pd.DataFrame:
    """Run Risk4 backtest with regime-dependent exposure scaling."""

    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        return pd.DataFrame()

    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    # Build regime lookup
    regime_lookup = {}
    if regime_df is not None and len(regime_df) > 0:
        for _, row in regime_df.iterrows():
            regime_lookup[row['date']] = row['vol_regime']

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    records = []
    w_old = pd.Series(dtype=float)

    for d in rebal_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.copy()
        bench_ret = float(universe["target"].mean())

        # Get regime exposure
        vol_regime = regime_lookup.get(d, "unknown")
        exposure = exposure_fn(vol_regime)

        if exposure <= 0:
            # Cash — no portfolio return, just benchmark
            records.append({
                "date": d,
                "port_ret_raw": 0.0,
                "bench_ret_raw": bench_ret,
                "vol_regime": vol_regime,
                "exposure": exposure,
            })
            w_old = pd.Series(dtype=float)
            continue

        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue

        picks = picks.drop_duplicates(subset="ticker", keep="first")
        picks = picks.set_index("ticker")

        # Weights proportional to alpha_z / vol_blend
        w_prop = picks["alpha_z"].clip(lower=0.0)
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
        sector_port_w = picks.reindex(w_prop.index).groupby("sector").apply(
            lambda g: w_prop.reindex(g.index).sum()
        )
        caps = sector_univ_w * sector_cap_mult

        for sec in sector_port_w.index:
            w_sec = sector_port_w[sec]
            cap = caps.get(sec, 0.0)
            if w_sec > cap and cap > 0:
                scale = cap / w_sec
                mask = picks.loc[w_prop.index, "sector"] == sec
                w_prop[mask] *= scale

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

        # Apply exposure scaling
        w_new = w_new * exposure

        w_old = w_new.copy()

        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue

        port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())

        records.append({
            "date": d,
            "port_ret_raw": port_ret,
            "bench_ret_raw": bench_ret,
            "vol_regime": vol_regime,
            "exposure": exposure,
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
# COMPARISON
# =============================================================================

def summarize_backtest(bt: pd.DataFrame, name: str, rebalance_every: int = 5) -> dict:
    """Compute summary metrics for a backtest result."""
    if bt.empty:
        return {"variant": name, "error": "empty"}

    ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], rebalance_every)
    eq = (1 + bt["port_ret"]).cumprod()
    mdd = max_drawdown(eq)

    # Regime-conditioned performance
    regime_stats = {}
    if "vol_regime" in bt.columns:
        for regime in bt["vol_regime"].unique():
            subset = bt[bt["vol_regime"] == regime]
            if len(subset) > 5:
                r, v, s = annualize(subset["port_ret"], rebalance_every)
                regime_stats[regime] = {"return": r, "sharpe": s, "n_periods": len(subset)}

    return {
        "variant": name,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "total_return": float(eq.iloc[-1] - 1) if len(eq) > 0 else 0,
        "n_periods": len(bt),
        "pct_invested": float((bt.get("exposure", 1.0) > 0).mean()),
        "regime_stats": regime_stats,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compare exposure rule variants using Risk4 backtester"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", default="alpha_ml_v2_tuned_clf")
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-11-01")
    parser.add_argument("--top-n", type=int, default=75)
    parser.add_argument("--target-vol", type=float, default=0.25)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("EXPOSURE VARIANT COMPARISON")
    logger.info("  Step 6 of Kairos Objective & Promotion Policy")
    logger.info("=" * 70)

    con = duckdb.connect(args.db, read_only=True)

    # Load data
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
    logger.info(f"  Loaded {len(df):,} rows")

    # Load regime history
    regime_df = con.execute("""
        SELECT date, vol_regime, trend_regime, regime
        FROM regime_history_academic
    """).fetchdf()
    logger.info(f"  Loaded {len(regime_df):,} regime rows")

    con.close()

    # Run all variants
    results = []
    for name, variant in VARIANTS.items():
        logger.info(f"\nRunning variant: {name} — {variant['description']}")
        bt = run_backtest_with_exposure(
            df.copy(), regime_df,
            exposure_fn=variant["exposure_fn"],
            top_n=args.top_n,
            rebalance_every=args.rebalance_every,
            target_vol=args.target_vol,
        )
        summary = summarize_backtest(bt, name, args.rebalance_every)
        results.append(summary)

        if "error" not in summary:
            logger.info(f"  Sharpe: {summary['sharpe']:.3f}  "
                        f"Return: {summary['ann_return']:.1%}  "
                        f"MaxDD: {summary['max_drawdown']:.1%}  "
                        f"Invested: {summary['pct_invested']:.0%}")

    # Print comparison table
    print(f"\n\n{'=' * 90}")
    print("  EXPOSURE VARIANT COMPARISON")
    print(f"{'=' * 90}")
    print(f"{'Variant':<20} {'Sharpe':>8} {'Ann Ret':>9} {'Ann Vol':>9} "
          f"{'Max DD':>8} {'Tot Ret':>9} {'Invested':>9}")
    print("-" * 90)

    for r in results:
        if "error" in r:
            print(f"{r['variant']:<20} {'ERROR':>8}")
            continue
        print(f"{r['variant']:<20} {r['sharpe']:>8.3f} {r['ann_return']:>9.1%} "
              f"{r['ann_vol']:>9.1%} {r['max_drawdown']:>8.1%} "
              f"{r['total_return']:>9.1%} {r['pct_invested']:>9.0%}")

    # Regime breakdown for each variant
    print(f"\n\n{'=' * 90}")
    print("  REGIME-CONDITIONED SHARPE BY VARIANT")
    print(f"{'=' * 90}")

    all_regimes = sorted(set(
        r for s in results if "regime_stats" in s
        for r in s.get("regime_stats", {}).keys()
    ))

    header = f"{'Variant':<20}"
    for reg in all_regimes:
        header += f" {reg:>12}"
    print(header)
    print("-" * 90)

    for s in results:
        if "error" in s:
            continue
        line = f"{s['variant']:<20}"
        for reg in all_regimes:
            rs = s.get("regime_stats", {}).get(reg, {})
            sharpe = rs.get("sharpe", float("nan"))
            line += f" {sharpe:>12.3f}" if not np.isnan(sharpe) else f" {'N/A':>12}"
        print(line)

    # Save
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        comparison_df = pd.DataFrame([
            {k: v for k, v in r.items() if k != "regime_stats"}
            for r in results
        ])
        comparison_df.to_csv(out / "exposure_variant_comparison.csv", index=False)
        logger.info(f"\nSaved comparison to: {out / 'exposure_variant_comparison.csv'}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
