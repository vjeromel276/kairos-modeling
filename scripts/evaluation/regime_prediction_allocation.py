#!/usr/bin/env python3
"""
regime_prediction_allocation.py
================================
Predict future regime states and use predictions to lead capital allocation.

Two-stage system:
    Stage 1: Predict what vol/trend regime will be in 1-2 weeks
    Stage 2: Use predicted regime to set allocation BEFORE the regime arrives

The regime is highly persistent (84% week-to-week for vol, 74% for trend),
and current cross-sectional volatility has IC 0.64 with next-week high_vol.
This means we can build a leading signal.

Prediction features (all market-level, computed cross-sectionally):
    - Current avg vol, vol change, vol acceleration
    - Vol dispersion (cross-sectional spread)
    - Avg returns at multiple horizons
    - Momentum breadth
    - Current regime state (one-hot)

Allocation strategies tested:
    - Predicted regime -> Kelly-informed allocation
    - Predicted regime -> CPPI with regime-adjusted floor
    - Predicted regime + drawdown -> hybrid

Usage:
    python scripts/evaluation/regime_prediction_allocation.py \
        --db data/kairos.duckdb \
        --output-dir outputs/evaluation/regime_prediction
"""

import argparse
import logging
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr

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
    return float((eq / run_max - 1.0).min())


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
# STAGE 1: BUILD REGIME PREDICTION FEATURES
# =============================================================================

def build_regime_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Build market-level features that predict future regime state.
    All features are cross-sectional aggregates — no lookahead.
    """

    logger.info("Building regime prediction features...")

    # Cross-sectional aggregates per date
    mkt = con.execute("""
        SELECT
            date,
            AVG(vol_blend) as avg_vol,
            STDDEV(vol_blend) as vol_dispersion,
            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY vol_blend) as vol_p90,
            PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY vol_blend) as vol_p10,
            AVG(ret_5d) as avg_ret_5d,
            AVG(ret_21d) as avg_ret_21d,
            STDDEV(ret_5d) as ret_5d_dispersion,
            AVG(beta_252d) as avg_beta,
            AVG(resid_vol_63d) as avg_resid_vol,
            AVG(mom_1m) as avg_mom_1m,
            AVG(mom_3m) as avg_mom_3m,
            -- Breadth: fraction of stocks with positive momentum
            AVG(CASE WHEN mom_1m > 0 THEN 1.0 ELSE 0.0 END) as breadth_1m,
            AVG(CASE WHEN mom_3m > 0 THEN 1.0 ELSE 0.0 END) as breadth_3m,
            COUNT(*) as n_stocks
        FROM feat_matrix_v2
        WHERE vol_blend IS NOT NULL
          AND date >= '2015-01-01'
        GROUP BY date
        ORDER BY date
    """).fetchdf()
    mkt['date'] = pd.to_datetime(mkt['date'])

    # Add lagged/change features
    mkt['vol_change_5d'] = mkt['avg_vol'] - mkt['avg_vol'].shift(5)
    mkt['vol_change_21d'] = mkt['avg_vol'] - mkt['avg_vol'].shift(21)
    mkt['vol_accel'] = mkt['vol_change_5d'] - mkt['vol_change_5d'].shift(5)
    mkt['vol_ratio_5_21'] = mkt['avg_vol'] / mkt['avg_vol'].rolling(21).mean()
    mkt['ret_5d_change'] = mkt['avg_ret_5d'] - mkt['avg_ret_5d'].shift(5)
    mkt['breadth_change_1m'] = mkt['breadth_1m'] - mkt['breadth_1m'].shift(5)
    mkt['dispersion_change'] = mkt['vol_dispersion'] - mkt['vol_dispersion'].shift(5)
    mkt['vol_range'] = mkt['vol_p90'] - mkt['vol_p10']

    # Join regime labels
    regime = con.execute("""
        SELECT date, vol_regime, trend_regime, regime
        FROM regime_history_academic
        ORDER BY date
    """).fetchdf()
    regime['date'] = pd.to_datetime(regime['date'])

    mkt = mkt.merge(regime, on='date', how='inner')

    # Current regime as features (one-hot)
    mkt['is_high_vol'] = (mkt['vol_regime'] == 'high_vol').astype(int)
    mkt['is_low_vol'] = (mkt['vol_regime'] == 'low_vol').astype(int)
    mkt['is_bear'] = (mkt['trend_regime'] == 'bear').astype(int)
    mkt['is_bull'] = (mkt['trend_regime'] == 'bull').astype(int)

    # Forward targets: regime in 1 week (5 trading days)
    mkt['vol_regime_1w'] = mkt['vol_regime'].shift(-5)
    mkt['trend_regime_1w'] = mkt['trend_regime'].shift(-5)

    # Numeric targets for XGBoost
    vol_map = {'low_vol': 0, 'normal_vol': 1, 'high_vol': 2}
    trend_map = {'bear': 0, 'neutral': 1, 'bull': 2}
    mkt['vol_target'] = mkt['vol_regime_1w'].map(vol_map)
    mkt['trend_target'] = mkt['trend_regime_1w'].map(trend_map)

    # Binary: will vol increase next week?
    mkt['vol_up_1w'] = (mkt['avg_vol'].shift(-5) > mkt['avg_vol']).astype(int)

    logger.info(f"  Built {len(mkt):,} rows of regime features")

    return mkt


REGIME_FEATURES = [
    'avg_vol', 'vol_dispersion', 'vol_p90', 'vol_p10', 'vol_range',
    'avg_ret_5d', 'avg_ret_21d', 'ret_5d_dispersion',
    'avg_beta', 'avg_resid_vol',
    'avg_mom_1m', 'avg_mom_3m',
    'breadth_1m', 'breadth_3m',
    'vol_change_5d', 'vol_change_21d', 'vol_accel', 'vol_ratio_5_21',
    'ret_5d_change', 'breadth_change_1m', 'dispersion_change',
    'is_high_vol', 'is_low_vol', 'is_bear', 'is_bull',
]


# =============================================================================
# STAGE 1: TRAIN REGIME PREDICTOR
# =============================================================================

def train_regime_predictor(mkt: pd.DataFrame):
    """
    Walk-forward train a regime predictor.
    Returns predictions for the full period using expanding-window training.
    """

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: REGIME PREDICTION MODEL")
    logger.info("=" * 70)

    mkt = mkt.dropna(subset=['vol_target']).copy()
    mkt['year'] = mkt['date'].dt.year

    test_years = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
    all_predictions = []
    cv_results = []

    for test_year in test_years:
        train = mkt[mkt['year'] < test_year]
        test = mkt[mkt['year'] == test_year]

        if len(train) < 100 or len(test) < 20:
            continue

        X_train = train[REGIME_FEATURES].copy()
        X_test = test[REGIME_FEATURES].copy()

        for col in REGIME_FEATURES:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)

        # Vol regime prediction (multiclass)
        y_train_vol = train['vol_target'].values
        model_vol = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=3,
            random_state=42, verbosity=0,
        )
        model_vol.fit(X_train, y_train_vol, verbose=False)

        vol_probs = model_vol.predict_proba(X_test)
        vol_pred = model_vol.predict(X_test)
        vol_actual = test['vol_target'].values
        vol_acc = (vol_pred == vol_actual).mean()

        # Trend regime prediction
        y_train_trend = train['trend_target'].values
        model_trend = xgb.XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softprob', num_class=3,
            random_state=42, verbosity=0,
        )
        model_trend.fit(X_train, y_train_trend, verbose=False)

        trend_probs = model_trend.predict_proba(X_test)
        trend_pred = model_trend.predict(X_test)
        trend_actual = test['trend_target'].values
        trend_acc = (trend_pred == trend_actual).mean()

        logger.info(f"  {test_year}: vol_acc={vol_acc:.1%}  trend_acc={trend_acc:.1%}")

        cv_results.append({
            'test_year': test_year,
            'vol_accuracy': vol_acc,
            'trend_accuracy': trend_acc,
        })

        # Store predictions
        pred_df = test[['date', 'vol_regime', 'trend_regime']].copy()
        pred_df['pred_vol'] = [['low_vol', 'normal_vol', 'high_vol'][v] for v in vol_pred]
        pred_df['pred_trend'] = [['bear', 'neutral', 'bull'][t] for t in trend_pred]
        pred_df['p_high_vol'] = vol_probs[:, 2]
        pred_df['p_low_vol'] = vol_probs[:, 0]
        pred_df['p_bear'] = trend_probs[:, 0]
        pred_df['p_bull'] = trend_probs[:, 2]

        # Risk score: higher = more dangerous (high vol + bear probability)
        pred_df['risk_score'] = pred_df['p_high_vol'] * 0.5 + pred_df['p_bear'] * 0.5

        # Opportunity score: higher = better (high vol probability, since our signal
        # is strongest there)
        pred_df['opportunity_score'] = pred_df['p_high_vol']

        all_predictions.append(pred_df)

    predictions = pd.concat(all_predictions, ignore_index=True)

    cv_df = pd.DataFrame(cv_results)
    logger.info(f"\n  Mean vol accuracy:   {cv_df['vol_accuracy'].mean():.1%}")
    logger.info(f"  Mean trend accuracy: {cv_df['trend_accuracy'].mean():.1%}")

    # Feature importance from last model
    logger.info(f"\n  Top 10 vol prediction features:")
    imp = pd.Series(model_vol.feature_importances_, index=REGIME_FEATURES)
    for feat, val in imp.nlargest(10).items():
        logger.info(f"    {feat:<25} {val:.4f}")

    return predictions, cv_df


# =============================================================================
# STAGE 2: ALLOCATION STRATEGIES USING PREDICTIONS
# =============================================================================

ALLOCATION_STRATEGIES = {
    'full_baseline': {
        'description': 'Always 100% (no prediction)',
        'fn': 'full',
    },
    'cppi_baseline': {
        'description': 'CPPI 85/5 (best from prior test, no prediction)',
        'fn': 'cppi',
        'params': {'floor_pct': 0.85, 'multiplier': 5.0},
    },
    'predicted_regime_alloc': {
        'description': 'Allocation based on predicted vol regime',
        'fn': 'pred_regime',
        'params': {
            # Allocations by PREDICTED vol regime (high_vol is our best regime)
            'high_vol': 1.0,     # Full — signal is strongest
            'normal_vol': 0.85,  # Slightly reduce
            'low_vol': 0.70,     # Reduce — signal weakest
        },
    },
    'risk_score_alloc': {
        'description': 'Scale by inverse risk score (continuous)',
        'fn': 'risk_score',
        'params': {
            'min_alloc': 0.50,
            'max_alloc': 1.0,
        },
    },
    'opportunity_tilt': {
        'description': 'Tilt allocation toward high-opportunity (high vol) periods',
        'fn': 'opportunity',
        'params': {
            'base_alloc': 0.80,
            'max_boost': 1.0,  # Up to 100%
        },
    },
    'cppi_plus_prediction': {
        'description': 'CPPI with regime-adjusted multiplier',
        'fn': 'cppi_pred',
        'params': {
            'floor_pct': 0.85,
            'base_multiplier': 5.0,
            'regime_mult': {
                'high_vol': 1.3,    # More aggressive in best regime
                'normal_vol': 1.0,
                'low_vol': 0.7,     # More conservative
            },
        },
    },
    'dd_plus_prediction': {
        'description': 'Drawdown scaling with predicted regime boost',
        'fn': 'dd_pred',
        'params': {
            'max_dd_threshold': -0.20,
            'min_exposure': 0.30,
            'regime_mult': {
                'high_vol': 1.2,
                'normal_vol': 1.0,
                'low_vol': 0.85,
            },
        },
    },
}


def compute_allocation(strategy_fn, equity, peak, pred_row, params):
    """Compute allocation for a given strategy."""

    dd = (equity / peak) - 1.0 if peak > 0 else 0.0

    if strategy_fn == 'full':
        return 1.0

    elif strategy_fn == 'cppi':
        floor = peak * params['floor_pct']
        cushion = (equity - floor) / equity if equity > 0 else 0
        if cushion <= 0:
            return 0.0
        return min(params['multiplier'] * cushion, 1.0)

    elif strategy_fn == 'pred_regime':
        pred_vol = pred_row.get('pred_vol', 'normal_vol')
        return params.get(pred_vol, 0.85)

    elif strategy_fn == 'risk_score':
        risk = pred_row.get('risk_score', 0.5)
        span = params['max_alloc'] - params['min_alloc']
        return params['max_alloc'] - risk * span

    elif strategy_fn == 'opportunity':
        opp = pred_row.get('opportunity_score', 0.33)
        return params['base_alloc'] + opp * (params['max_boost'] - params['base_alloc'])

    elif strategy_fn == 'cppi_pred':
        pred_vol = pred_row.get('pred_vol', 'normal_vol')
        regime_m = params['regime_mult'].get(pred_vol, 1.0)
        mult = params['base_multiplier'] * regime_m
        floor = peak * params['floor_pct']
        cushion = (equity - floor) / equity if equity > 0 else 0
        if cushion <= 0:
            return 0.0
        return min(mult * cushion, 1.0)

    elif strategy_fn == 'dd_pred':
        pred_vol = pred_row.get('pred_vol', 'normal_vol')
        regime_m = params['regime_mult'].get(pred_vol, 1.0)
        max_dd = params['max_dd_threshold']
        min_exp = params['min_exposure']
        if dd >= 0:
            base = 1.0
        else:
            frac = min(dd / max_dd, 1.0)
            base = 1.0 - frac * (1.0 - min_exp)
        return min(base * regime_m, 1.0)

    return 1.0


# =============================================================================
# PORTFOLIO BACKTEST WITH PREDICTED ALLOCATION
# =============================================================================

def run_allocation_backtest(
    stock_df: pd.DataFrame,
    predictions: pd.DataFrame,
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
) -> pd.DataFrame:
    """Run Risk4 backtest with prediction-driven allocation."""

    stock_df = stock_df.dropna(subset=["sector"])
    stock_df = stock_df[stock_df["adv_20"] >= adv_thresh]
    stock_df["alpha_z"] = stock_df.groupby("date")["alpha"].transform(safe_z)
    stock_df["alpha_z"] = stock_df["alpha_z"].clip(-3.0, 3.0)

    # Build prediction lookup
    pred_lookup = {}
    for _, row in predictions.iterrows():
        pred_lookup[row['date']] = row.to_dict()

    dates = sorted(stock_df["date"].unique())
    rebal_dates = dates[::rebalance_every]

    strategy_fn = strategy_config['fn']
    params = strategy_config.get('params', {})

    records = []
    w_old = pd.Series(dtype=float)
    equity = 1.0
    peak = 1.0

    for d in rebal_dates:
        day = stock_df[stock_df["date"] == d].copy()
        if day.empty:
            continue

        universe = day.copy()
        bench_ret = float(universe["target"].mean())

        # Get prediction for this date
        pred_row = pred_lookup.get(d, {})

        # Compute allocation
        allocation = compute_allocation(strategy_fn, equity, peak, pred_row, params)
        allocation = max(0.0, min(1.0, allocation))

        if allocation <= 0.001:
            equity_before = equity
            records.append({
                "date": d, "port_ret_raw": 0.0, "bench_ret_raw": bench_ret,
                "allocation": 0.0,
                "pred_vol": pred_row.get('pred_vol', 'unknown'),
                "actual_vol": pred_row.get('vol_regime', 'unknown'),
            })
            continue

        # Standard Risk4 construction
        picks = day.sort_values("alpha_z", ascending=False).head(top_n).copy()
        if picks.empty:
            continue
        picks = picks.drop_duplicates(subset="ticker", keep="first").set_index("ticker")

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
        sector_univ_w = sector_counts / float(len(universe))
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
            all_t = pd.Index(w_old.index).union(w_prop.index).unique()
            wo = w_old.reindex(all_t).fillna(0.0)
            wp = w_prop.reindex(all_t).fillna(0.0)
            w_pre = (1 - lambda_tc) * wo + lambda_tc * wp
            to = float((w_pre - wo).abs().sum())
            if to > turnover_cap:
                w_new_f = wo + (turnover_cap / to) * (w_pre - wo)
            else:
                w_new_f = w_pre
            if w_new_f.sum() <= 0:
                w_new_f = wo
            w_new_f = w_new_f / w_new_f.sum()
            w_new = w_new_f.reindex(w_prop.index).fillna(0.0)

        w_new = w_new[~w_new.index.duplicated(keep="first")]
        w_old = w_new.copy()

        common = picks.index.intersection(w_new.index)
        if len(common) == 0:
            continue

        raw_ret = float((picks.loc[common, "target"] * w_new.loc[common]).sum())
        scaled_ret = raw_ret * allocation

        equity *= (1 + scaled_ret)
        peak = max(peak, equity)

        records.append({
            "date": d, "port_ret_raw": scaled_ret, "bench_ret_raw": bench_ret,
            "allocation": allocation,
            "pred_vol": pred_row.get('pred_vol', 'unknown'),
            "actual_vol": pred_row.get('vol_regime', 'unknown'),
        })

    bt = pd.DataFrame.from_records(records).sort_values("date").reset_index(drop=True)
    if bt.empty:
        return bt

    raw = bt["port_ret_raw"]
    _, raw_vol, _ = annualize(raw, rebalance_every)
    scale = target_vol / raw_vol if raw_vol > 0 else 1.0
    bt["port_ret"] = bt["port_ret_raw"] * scale
    bt["bench_ret"] = bt["bench_ret_raw"]

    return bt


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Regime prediction + capital allocation"
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
    logger.info("REGIME PREDICTION + CAPITAL ALLOCATION")
    logger.info("  Leading signal for allocation decisions")
    logger.info("=" * 70)

    con = duckdb.connect(args.db, read_only=True)

    # Stage 1: Build regime prediction
    mkt = build_regime_features(con)
    predictions, cv_df = train_regime_predictor(mkt)

    # Load stock data for backtest
    logger.info("\nLoading stock data...")
    stock_df = con.execute(f"""
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
    logger.info(f"  Loaded {len(stock_df):,} rows")

    # Stage 2: Run allocation strategies
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: ALLOCATION BACKTEST WITH PREDICTIONS")
    logger.info("=" * 70)

    results = []
    for name, config in ALLOCATION_STRATEGIES.items():
        logger.info(f"\n  Running: {name} — {config['description']}")

        bt = run_allocation_backtest(
            stock_df.copy(), predictions,
            strategy_name=name,
            strategy_config=config,
            target_vol=args.target_vol,
        )

        if bt.empty:
            logger.warning(f"    Empty backtest for {name}")
            continue

        ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], 5)
        eq = (1 + bt["port_ret"]).cumprod()
        mdd = max_drawdown(eq)
        total_ret = float(eq.iloc[-1] - 1)
        cal = ann_ret / abs(mdd) if mdd != 0 else 0
        avg_alloc = bt["allocation"].mean()

        results.append({
            "strategy": name,
            "description": config["description"],
            "sharpe": sharpe,
            "ann_return": ann_ret,
            "max_drawdown": mdd,
            "calmar": cal,
            "total_return": total_ret,
            "avg_allocation": avg_alloc,
        })

        logger.info(f"    Sharpe={sharpe:.3f}  Return={ann_ret:.1%}  "
                    f"MaxDD={mdd:.1%}  Calmar={cal:.2f}")

    results_df = pd.DataFrame(results).sort_values("calmar", ascending=False)

    # Print results
    print(f"\n\n{'=' * 120}")
    print("  REGIME PREDICTION + ALLOCATION COMPARISON (sorted by Calmar)")
    print(f"{'=' * 120}")
    print(f"{'Strategy':<25} {'Sharpe':>7} {'AnnRet':>8} {'MaxDD':>7} "
          f"{'Calmar':>7} {'TotRet':>9} {'AvgAlloc':>9}")
    print("-" * 120)

    for _, r in results_df.iterrows():
        print(f"{r['strategy']:<25} {r['sharpe']:>7.3f} {r['ann_return']:>8.1%} "
              f"{r['max_drawdown']:>7.1%} {r['calmar']:>7.2f} "
              f"{r['total_return']:>9.1%} {r['avg_allocation']:>9.0%}")

    baseline = results_df[results_df['strategy'] == 'full_baseline']
    best = results_df.iloc[0]
    if len(baseline) > 0:
        b = baseline.iloc[0]
        print(f"\n  BASELINE:  Sharpe={b['sharpe']:.3f}  Return={b['ann_return']:.1%}  "
              f"MaxDD={b['max_drawdown']:.1%}  Calmar={b['calmar']:.2f}")
        print(f"  BEST:      Sharpe={best['sharpe']:.3f}  Return={best['ann_return']:.1%}  "
              f"MaxDD={best['max_drawdown']:.1%}  Calmar={best['calmar']:.2f}")
        print(f"  Calmar gain: {best['calmar'] - b['calmar']:+.2f}")

    if args.output_dir:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out / "regime_prediction_allocation.csv", index=False)
        cv_df.to_csv(out / "regime_prediction_cv.csv", index=False)
        predictions.to_csv(out / "regime_predictions.csv", index=False)
        logger.info(f"\nSaved to: {out}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
