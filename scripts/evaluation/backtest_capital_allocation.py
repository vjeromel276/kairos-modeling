#!/usr/bin/env python3
"""
backtest_capital_allocation.py
===============================
Backtest capital allocation strategies that protect compounding during
drawdowns without giving up too much return.

Tests three families of allocation:
    1. Drawdown-aware scaling: reduce exposure proportionally as drawdown deepens
    2. CPPI-inspired: maintain a floor, invest a multiple of the cushion
    3. Hybrid: drawdown scaling with regime tilt (increase aggression in best regimes)

All strategies use the same v2_tuned long-only signal with Risk4 portfolio
construction. The variable is HOW MUCH capital is deployed, not WHAT is held.

Usage:
    python scripts/evaluation/backtest_capital_allocation.py \
        --db data/kairos.duckdb \
        --output-dir outputs/evaluation/capital_allocation
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


def calmar(ann_ret: float, mdd: float) -> float:
    return ann_ret / abs(mdd) if mdd != 0 else 0.0


# =============================================================================
# ALLOCATION STRATEGIES
# =============================================================================

def alloc_full(equity_curve, peak, params):
    """Always 100% invested."""
    return 1.0


def alloc_drawdown_linear(equity_curve, peak, params):
    """
    Linear drawdown scaling.
    At 0% DD: full exposure. At max_dd_threshold: min_exposure.
    """
    dd = (equity_curve[-1] / peak) - 1.0
    max_dd = params['max_dd_threshold']  # e.g., -0.20
    min_exp = params['min_exposure']      # e.g., 0.30

    if dd >= 0:
        return 1.0

    # Linear interpolation: 0% DD -> 1.0, max_dd -> min_exposure
    frac = dd / max_dd  # 0 to 1 as drawdown deepens
    frac = min(frac, 1.0)
    return 1.0 - frac * (1.0 - min_exp)


def alloc_drawdown_convex(equity_curve, peak, params):
    """
    Convex drawdown scaling (cuts faster as DD deepens).
    Uses power function for more aggressive protection at deeper drawdowns.
    """
    dd = (equity_curve[-1] / peak) - 1.0
    max_dd = params['max_dd_threshold']
    min_exp = params['min_exposure']
    power = params.get('power', 2.0)

    if dd >= 0:
        return 1.0

    frac = min(dd / max_dd, 1.0)
    return 1.0 - (frac ** power) * (1.0 - min_exp)


def alloc_cppi(equity_curve, peak, params):
    """
    CPPI-inspired: exposure = multiplier * (value - floor) / value.
    Floor is set as (1 - max_loss) * peak.
    """
    value = equity_curve[-1]
    floor_pct = params['floor_pct']      # e.g., 0.85 (won't lose more than 15% from peak)
    multiplier = params['multiplier']     # e.g., 3.0
    floor = peak * floor_pct
    cushion = (value - floor) / value

    if cushion <= 0:
        return 0.0  # At or below floor — go to cash

    exposure = min(multiplier * cushion, 1.0)
    return max(exposure, 0.0)


def alloc_drawdown_with_regime(equity_curve, peak, params, vol_regime=None):
    """
    Drawdown scaling + regime tilt.
    In strong regimes (high_vol), allow more aggressive recovery.
    """
    base = alloc_drawdown_linear(equity_curve, peak, params)

    regime_boost = params.get('regime_boosts', {})
    boost = regime_boost.get(vol_regime, 1.0)

    return min(base * boost, 1.0)


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    'full_exposure': {
        'fn': 'full',
        'params': {},
        'description': 'Always 100% invested (baseline)',
    },
    'dd_linear_20_30': {
        'fn': 'dd_linear',
        'params': {'max_dd_threshold': -0.20, 'min_exposure': 0.30},
        'description': 'Linear: 100% at 0% DD, 30% at -20% DD',
    },
    'dd_linear_15_50': {
        'fn': 'dd_linear',
        'params': {'max_dd_threshold': -0.15, 'min_exposure': 0.50},
        'description': 'Linear: 100% at 0% DD, 50% at -15% DD (conservative)',
    },
    'dd_linear_25_20': {
        'fn': 'dd_linear',
        'params': {'max_dd_threshold': -0.25, 'min_exposure': 0.20},
        'description': 'Linear: 100% at 0% DD, 20% at -25% DD (patient)',
    },
    'dd_convex_20_30': {
        'fn': 'dd_convex',
        'params': {'max_dd_threshold': -0.20, 'min_exposure': 0.30, 'power': 2.0},
        'description': 'Convex: stays fuller longer, cuts hard near -20%',
    },
    'dd_convex_20_30_p3': {
        'fn': 'dd_convex',
        'params': {'max_dd_threshold': -0.20, 'min_exposure': 0.30, 'power': 3.0},
        'description': 'Convex p=3: very patient, aggressive cut at deep DD',
    },
    'cppi_85_3': {
        'fn': 'cppi',
        'params': {'floor_pct': 0.85, 'multiplier': 3.0},
        'description': 'CPPI: 15% max loss from peak, 3x multiplier',
    },
    'cppi_85_5': {
        'fn': 'cppi',
        'params': {'floor_pct': 0.85, 'multiplier': 5.0},
        'description': 'CPPI: 15% max loss from peak, 5x multiplier (aggressive)',
    },
    'cppi_85_667_PROD': {
        'fn': 'cppi',
        'params': {'floor_pct': 0.85, 'multiplier': 6.67},
        'description': 'CPPI: 85% floor, 6.67x (PRODUCTION config — full expo at peak)',
    },
    'cppi_80_3': {
        'fn': 'cppi',
        'params': {'floor_pct': 0.80, 'multiplier': 3.0},
        'description': 'CPPI: 20% max loss from peak, 3x multiplier',
    },
    'cppi_90_3': {
        'fn': 'cppi',
        'params': {'floor_pct': 0.90, 'multiplier': 3.0},
        'description': 'CPPI: 10% max loss from peak, 3x multiplier (tight)',
    },
    'dd_regime_hybrid': {
        'fn': 'dd_regime',
        'params': {
            'max_dd_threshold': -0.20,
            'min_exposure': 0.30,
            'regime_boosts': {
                'high_vol': 1.3,     # Best regime — push harder
                'normal_vol': 1.0,   # Normal
                'low_vol': 0.9,      # Weakest regime — slightly less
            }
        },
        'description': 'DD scaling + regime tilt (boost in high-vol)',
    },
}


# =============================================================================
# RISK4 BACKTEST WITH DYNAMIC ALLOCATION
# =============================================================================

def run_backtest(
    df: pd.DataFrame,
    regime_df: pd.DataFrame,
    strategy_name: str,
    strategy_config: dict,
    top_n: int = 75,
    rebalance_every: int = 5,
    target_vol: float = 0.25,
    adv_thresh: float = 2_000_000,
    max_stock_w: float = 0.03,
    sector_cap_mult: float = 2.0,
    lambda_tc: float = 0.5,
    turnover_cap: float = 0.30,
    spread_bps: float = 0.0,
    impact_coef: float = 0.0,
    portfolio_value: float = 100_000,
) -> pd.DataFrame:
    """Run Risk4 backtest with dynamic capital allocation.

    Transaction cost model (set spread_bps > 0 to enable):
      - Per-period turnover measured in POSITION space (weight × allocation).
        Accounts for both shape changes and CPPI allocation changes.
      - Flat round-trip spread cost: `spread_bps × turnover` per period.
      - Optional linear market impact: `impact_coef × trade_value / adv_20`
        summed per traded ticker (requires portfolio_value assumption).
    """

    df = df.dropna(subset=["sector"])
    df = df[df["adv_20"] >= adv_thresh]
    if df.empty:
        return pd.DataFrame()

    df["alpha_z"] = df.groupby("date")["alpha"].transform(safe_z)
    df["alpha_z"] = df["alpha_z"].clip(-3.0, 3.0)

    # Build regime lookup
    regime_lookup = {}
    if regime_df is not None:
        for _, row in regime_df.iterrows():
            regime_lookup[row['date']] = row['vol_regime']

    dates = sorted(df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    fn_name = strategy_config['fn']
    params = strategy_config['params']

    records = []
    w_old = pd.Series(dtype=float)
    prev_allocation = 0.0  # starts flat — first period has full build-up turnover
    position_prev = pd.Series(dtype=float)
    equity = 1.0
    peak = 1.0
    equity_history = [1.0]

    for d in rebal_dates:
        day = df[df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.copy()
        bench_ret = float(universe["target"].mean())

        # Determine allocation
        vol_regime = regime_lookup.get(d, 'unknown')

        if fn_name == 'full':
            allocation = alloc_full(equity_history, peak, params)
        elif fn_name == 'dd_linear':
            allocation = alloc_drawdown_linear(equity_history, peak, params)
        elif fn_name == 'dd_convex':
            allocation = alloc_drawdown_convex(equity_history, peak, params)
        elif fn_name == 'cppi':
            allocation = alloc_cppi(equity_history, peak, params)
        elif fn_name == 'dd_regime':
            allocation = alloc_drawdown_with_regime(
                equity_history, peak, params, vol_regime
            )
        else:
            allocation = 1.0

        if allocation <= 0.001:
            # Cash — sell everything
            turnover = float(position_prev.abs().sum())
            cost = turnover * (spread_bps / 10000.0)
            net_ret = -cost
            equity *= (1 + net_ret)
            peak = max(peak, equity)
            equity_history.append(equity)
            records.append({
                "date": d,
                "port_ret_raw": 0.0,
                "port_ret_net": net_ret,
                "bench_ret_raw": bench_ret,
                "allocation": 0.0,
                "vol_regime": vol_regime,
                "turnover": turnover,
                "cost": cost,
            })
            w_old = pd.Series(dtype=float)
            prev_allocation = 0.0
            position_prev = pd.Series(dtype=float)
            continue

        # Standard Risk4 portfolio construction
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue
        picks = picks.drop_duplicates(subset="ticker", keep="first")
        picks = picks.set_index("ticker")

        w_prop = picks["alpha_z"].clip(lower=0.0)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / picks["vol_blend"].replace(0, np.nan)
        w_prop = w_prop.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if w_prop.sum() == 0:
            continue
        w_prop = w_prop / w_prop.sum()

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

        # Apply allocation scaling
        raw_port_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        scaled_ret = raw_port_ret * allocation

        # --- Transaction costs ---
        # position_new = w_new × allocation (actual deployed weights).
        # turnover = L1 distance between positions (captures both shape
        # changes and allocation changes). First period: builds from 0.
        position_new = (w_new * allocation).fillna(0.0)
        all_tick = pd.Index(position_prev.index).union(position_new.index).unique()
        pn = position_new.reindex(all_tick).fillna(0.0)
        pp = position_prev.reindex(all_tick).fillna(0.0)
        delta = (pn - pp).abs()
        turnover = float(delta.sum())

        spread_cost = turnover * (spread_bps / 10000.0)
        impact_cost = 0.0
        if impact_coef > 0.0:
            # Participation rate per ticker (trade_value / adv), bounded to [0, 1]:
            #   participation = delta * portfolio_value / adv_20
            # Impact as fraction of trade value = impact_coef * participation
            # Cost as fraction of portfolio = delta * impact_coef * participation
            #                               = impact_coef * delta^2 * pv / adv
            # (Scales with pv — larger books pay more in impact.)
            adv_series = picks.reindex(all_tick)["adv_20"].fillna(np.inf)
            participation = ((delta * portfolio_value) / adv_series).clip(
                lower=0, upper=1
            )
            impact_cost = float((impact_coef * delta * participation).sum())

        total_cost = spread_cost + impact_cost
        net_ret = scaled_ret - total_cost

        # Update equity tracking (net)
        equity *= (1 + net_ret)
        peak = max(peak, equity)
        equity_history.append(equity)

        records.append({
            "date": d,
            "port_ret_raw": scaled_ret,
            "port_ret_net": net_ret,
            "bench_ret_raw": bench_ret,
            "allocation": allocation,
            "vol_regime": vol_regime,
            "turnover": turnover,
            "cost": total_cost,
        })

        prev_allocation = allocation
        position_prev = position_new

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt

    # Vol targeting on the raw (pre-cost, pre-allocation-scaling) returns
    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0

    # Scaling leverages positions, so cost scales linearly with leverage too.
    bt["port_ret_gross"] = bt["port_ret_raw"] * scale
    bt["cost_scaled"] = bt["cost"] * scale
    bt["port_ret"] = bt["port_ret_net"] * scale        # net column kept as primary
    bt["bench_ret"] = bt["bench_ret_raw"]
    bt["active_ret"] = bt["port_ret"] - bt["bench_ret"]

    return bt


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Backtest capital allocation strategies for drawdown protection"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--alpha-column", default="alpha_ml_v2_tuned_clf")
    parser.add_argument("--target-column", default="ret_5d_f")
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2025-11-01")
    parser.add_argument("--target-vol", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--spread-bps", type=float, default=0.0,
                        help="Round-trip spread cost in bps (e.g., 15). Default 0 = gross.")
    parser.add_argument("--impact-coef", type=float, default=0.0,
                        help="Linear market impact coefficient (0 disables).")
    parser.add_argument("--portfolio-value", type=float, default=100_000,
                        help="Portfolio size in $ (for impact model only)")
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("CAPITAL ALLOCATION STRATEGY COMPARISON")
    logger.info("  How to protect compounding without giving up returns")
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

    regime_df = con.execute("""
        SELECT date, vol_regime FROM regime_history_academic
    """).fetchdf()
    con.close()

    logger.info(f"  Loaded {len(df):,} rows, {len(regime_df):,} regime rows")

    # Run all strategies
    results = []
    for name, config in STRATEGIES.items():
        logger.info(f"\n  Running: {name} — {config['description']}")

        bt = run_backtest(
            df.copy(), regime_df,
            strategy_name=name,
            strategy_config=config,
            target_vol=args.target_vol,
            spread_bps=args.spread_bps,
            impact_coef=args.impact_coef,
            portfolio_value=args.portfolio_value,
        )

        if bt.empty:
            continue

        ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], 5)
        eq = (1 + bt["port_ret"]).cumprod()
        mdd = max_drawdown(eq)
        total_ret = float(eq.iloc[-1] - 1)
        cal = calmar(ann_ret, mdd)
        avg_alloc = bt["allocation"].mean()
        min_alloc = bt["allocation"].min()

        # Gross metrics for comparison when costs are applied
        gross_col = "port_ret_gross" if "port_ret_gross" in bt.columns else "port_ret"
        ann_ret_g, _, sharpe_g = annualize(bt[gross_col], 5)
        eq_g = (1 + bt[gross_col]).cumprod()
        cal_g = calmar(ann_ret_g, max_drawdown(eq_g))
        annual_turnover = bt["turnover"].sum() / max(1, len(bt)) * (252 / 5) \
            if "turnover" in bt.columns else 0.0
        annual_cost = bt["cost_scaled"].sum() / max(1, len(bt)) * (252 / 5) \
            if "cost_scaled" in bt.columns else 0.0

        results.append({
            "strategy": name,
            "description": config["description"],
            "sharpe": sharpe,
            "ann_return": ann_ret,
            "ann_vol": ann_vol,
            "max_drawdown": mdd,
            "calmar": cal,
            "total_return": total_ret,
            "avg_allocation": avg_alloc,
            "min_allocation": min_alloc,
            "sharpe_gross": sharpe_g,
            "ann_return_gross": ann_ret_g,
            "calmar_gross": cal_g,
            "annual_turnover": annual_turnover,
            "annual_cost": annual_cost,
        })

        logger.info(f"    Sharpe={sharpe:.3f}  Return={ann_ret:.1%}  "
                    f"MaxDD={mdd:.1%}  Calmar={cal:.2f}  AvgAlloc={avg_alloc:.0%}  "
                    f"AnnTurnover={annual_turnover:.1f}  AnnCost={annual_cost:.2%}")

    results_df = pd.DataFrame(results)

    # Sort by Calmar (the metric that answers the question)
    results_df = results_df.sort_values("calmar", ascending=False)

    print(f"\n\n{'=' * 120}")
    print("  CAPITAL ALLOCATION COMPARISON (sorted by Calmar = Return / MaxDrawdown)")
    print(f"{'=' * 120}")
    print(f"{'Strategy':<22} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} "
          f"{'Calmar':>7} {'TotRet':>9} {'AvgAlloc':>9} {'MinAlloc':>9}")
    print("-" * 120)

    for _, r in results_df.iterrows():
        print(f"{r['strategy']:<22} {r['sharpe']:>7.3f} {r['ann_return']:>8.1%} "
              f"{r['max_drawdown']:>7.1%} {r['calmar']:>7.2f} "
              f"{r['total_return']:>9.1%} {r['avg_allocation']:>9.0%} "
              f"{r['min_allocation']:>9.0%}")

    # Compare best vs baseline
    baseline = results_df[results_df['strategy'] == 'full_exposure']
    best = results_df.iloc[0]

    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f"\n  BASELINE: {b['strategy']}")
        print(f"    Sharpe={b['sharpe']:.3f}  Return={b['ann_return']:.1%}  "
              f"MaxDD={b['max_drawdown']:.1%}  Calmar={b['calmar']:.2f}")
        print(f"\n  BEST:     {best['strategy']}")
        print(f"    Sharpe={best['sharpe']:.3f}  Return={best['ann_return']:.1%}  "
              f"MaxDD={best['max_drawdown']:.1%}  Calmar={best['calmar']:.2f}")
        print(f"\n  TRADEOFF:")
        print(f"    Calmar improvement: {best['calmar'] - b['calmar']:+.2f}")
        print(f"    Return cost:        {best['ann_return'] - b['ann_return']:+.1%}")
        print(f"    Drawdown saved:     {best['max_drawdown'] - b['max_drawdown']:+.1%}")

    # Save
    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "capital_allocation_comparison.csv", index=False)
        logger.info(f"\nSaved to: {out}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
