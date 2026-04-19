#!/usr/bin/env python3
"""
train_ensemble_stacker.py
=========================
Walk-forward meta-stacker over the five existing alpha signals.

Path A (scoping, leaky): uses production-stored predictions directly, so
base models technically saw the test year during final training. Treat
absolute IC as upward-biased; treat RELATIVE lift (ensemble vs best single
signal) as the real measurement.

Inputs (from data/kairos.duckdb):
  - feat_matrix_v2.alpha_ml_v2_tuned_clf
  - feat_matrix_v2.alpha_ml_v2_tuned_reg
  - feat_matrix_v2.alpha_composite_v8
  - feat_matrix_v2.alpha_composite_v33_regime
  - feat_alpha_ml_xgb_v3_neutral.alpha_ml_v3_neutral_clf
  - feat_targets.ret_5d_f               (target)

Meta-model variants evaluated per fold:
  equal_weight   — z-score inputs per date, average
  ridge          — sklearn Ridge, alpha tuned via inner-year validation
  xgb_thin       — XGBRegressor (max_depth=3, n_estimators=50, lr=0.1)

Metrics per test year:
  pooled_spearman_ic       — matches v2_tuned CV methodology
  per_date_spearman_ic     — matches Temporimutator evaluation
  top_bottom_quintile_bps  — long-short portfolio spread (bps/wk)

Usage:
  python scripts/ml/train_ensemble_stacker.py --db data/kairos.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ensemble_stacker")

ALL_FEATURES = [
    "alpha_ml_v2_tuned_clf",
    "alpha_ml_v2_tuned_reg",
    "alpha_ml_v3_neutral_clf",
    "alpha_composite_v8",
    "alpha_composite_v33_regime",
]
FEATURES = ALL_FEATURES  # overridden by --features at runtime
TARGET = "ret_5d_f"

CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5
DATA_START = "2015-01-01"

RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]


def load_joined(con: duckdb.DuckDBPyConnection, use_oos: bool) -> pd.DataFrame:
    """
    When use_oos=True, v2_tuned signals come from feat_alpha_ml_xgb_v2_tuned_oos
    (walk-forward regenerated, clean OOS per test year). Rule-based signals
    (v8, v33_regime) are always loaded from feat_matrix_v2 since they don't
    have a training process that could leak.
    """
    log.info("Loading joined signal frame (use_oos=%s)...", use_oos)
    if use_oos:
        q = f"""
            SELECT t.ticker, t.date,
                   t.{TARGET},
                   oos.alpha_ml_v2_tuned_clf_oos AS alpha_ml_v2_tuned_clf,
                   oos.alpha_ml_v2_tuned_reg_oos AS alpha_ml_v2_tuned_reg,
                   m.alpha_composite_v8,
                   m.alpha_composite_v33_regime,
                   v3.alpha_ml_v3_neutral_clf
            FROM feat_targets t
            JOIN feat_alpha_ml_xgb_v2_tuned_oos oos
              ON oos.ticker = t.ticker AND oos.date = t.date
            JOIN feat_matrix_v2 m
              ON m.ticker = t.ticker AND m.date = t.date
            LEFT JOIN feat_alpha_ml_xgb_v3_neutral v3
              ON v3.ticker = t.ticker AND v3.date = t.date
            WHERE t.date >= DATE '{DATA_START}'
              AND t.{TARGET} IS NOT NULL
        """
    else:
        q = f"""
            SELECT t.ticker, t.date,
                   t.{TARGET},
                   m.alpha_ml_v2_tuned_clf,
                   m.alpha_ml_v2_tuned_reg,
                   m.alpha_composite_v8,
                   m.alpha_composite_v33_regime,
                   v3.alpha_ml_v3_neutral_clf
            FROM feat_targets t
            JOIN feat_matrix_v2 m
              ON m.ticker = t.ticker AND m.date = t.date
            LEFT JOIN feat_alpha_ml_xgb_v3_neutral v3
              ON v3.ticker = t.ticker AND v3.date = t.date
            WHERE t.date >= DATE '{DATA_START}'
              AND t.{TARGET} IS NOT NULL
        """
    df = con.execute(q).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    log.info("  rows=%d  tickers=%d  dates %s → %s",
             len(df), df["ticker"].nunique(), df["date"].min().date(), df["date"].max().date())
    return df


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby("date", sort=False)
    for c in cols:
        mean = grouped[c].transform("mean")
        std = grouped[c].transform("std").replace(0, np.nan)
        df[c] = ((df[c] - mean) / std).fillna(0.0)
    return df


def purged_train_test_split(df: pd.DataFrame, test_year: int, purge_days: int):
    """Train = (year < test_year) AND (date < max_train_date - purge_days).
       Test  = (year == test_year)."""
    test = df[df["year"] == test_year]
    train = df[df["year"] < test_year]
    if train.empty or test.empty:
        return None, None
    train_dates = sorted(train["date"].unique())
    if len(train_dates) <= purge_days:
        return None, None
    purge_cut = train_dates[-(purge_days + 1)]
    train = train[train["date"] <= purge_cut]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def pooled_ic(signal: np.ndarray, ret: np.ndarray) -> float:
    mask = ~(np.isnan(signal) | np.isnan(ret))
    if mask.sum() < 10:
        return float("nan")
    ic, _ = spearmanr(signal[mask], ret[mask])
    return float(ic)


def per_date_ic(signal: np.ndarray, ret: np.ndarray, dates: np.ndarray) -> dict:
    df = pd.DataFrame({"s": signal, "r": ret, "d": dates})
    vals = []
    for _, g in df.groupby("d"):
        if len(g) < 10:
            continue
        if g["s"].std() == 0 or g["r"].std() == 0:
            continue
        v, _ = spearmanr(g["s"], g["r"])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "n_dates": 0}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
            "n_dates": len(vals)}


def quintile_spread_bps(signal: np.ndarray, ret: np.ndarray, dates: np.ndarray) -> float:
    """Per date: top-quintile mean - bottom-quintile mean return. Then mean across dates."""
    df = pd.DataFrame({"s": signal, "r": ret, "d": dates})
    spreads = []
    for _, g in df.groupby("d"):
        if len(g) < 25:
            continue
        q = g["s"].quantile([0.2, 0.8]).to_numpy()
        top = g[g["s"] >= q[1]]["r"].mean()
        bot = g[g["s"] <= q[0]]["r"].mean()
        spreads.append(top - bot)
    return float(np.mean(spreads) * 1e4) if spreads else float("nan")  # bps


def fit_and_predict(variant: str, X_tr, y_tr, X_te, ridge_inner_val=None) -> np.ndarray:
    if variant == "equal_weight":
        # Inputs are already per-date z-scored in the calling code
        return X_te.mean(axis=1).to_numpy()
    if variant == "ridge":
        # inner alpha selection: try each alpha on ridge_inner_val holdout
        if ridge_inner_val is None or len(ridge_inner_val[0]) == 0:
            best_alpha = 1.0
        else:
            Xiv, yiv = ridge_inner_val
            best_alpha, best_ic = 1.0, -np.inf
            for a in RIDGE_ALPHAS:
                r = Ridge(alpha=a).fit(X_tr, y_tr)
                p = r.predict(Xiv)
                ic = pooled_ic(p, yiv)
                if ic > best_ic:
                    best_ic, best_alpha = ic, a
        model = Ridge(alpha=best_alpha).fit(X_tr, y_tr)
        return model.predict(X_te)
    if variant == "xgb_thin":
        model = xgb.XGBRegressor(
            max_depth=3, n_estimators=50, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            reg_lambda=1.0, random_state=42, n_jobs=-1, tree_method="hist",
        )
        model.fit(X_tr, y_tr, verbose=False)
        return model.predict(X_te)
    raise ValueError(variant)


def best_single_signal_ic(test_df: pd.DataFrame) -> tuple[str, float]:
    best_sig, best_ic = None, -np.inf
    for f in FEATURES:
        ic = pooled_ic(test_df[f].to_numpy(), test_df[TARGET].to_numpy())
        if ic > best_ic:
            best_ic, best_sig = ic, f
    return best_sig, best_ic


def evaluate(name: str, signal: np.ndarray, test_df: pd.DataFrame) -> dict:
    ret = test_df[TARGET].to_numpy()
    dates = test_df["date"].to_numpy()
    pd_ic = per_date_ic(signal, ret, dates)
    return {
        "variant": name,
        "pooled_ic": pooled_ic(signal, ret),
        "per_date_ic_mean": pd_ic["mean"],
        "per_date_ic_std": pd_ic["std"],
        "per_date_n_dates": pd_ic["n_dates"],
        "quintile_spread_bps": quintile_spread_bps(signal, ret, dates),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Ensemble stacker over 5 alpha signals")
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("scripts/ml/outputs"))
    p.add_argument("--mlflow-uri", default="http://localhost:5000")
    p.add_argument("--features", default=",".join(ALL_FEATURES),
                   help="Comma-separated subset of alpha signal columns to use as inputs")
    p.add_argument("--run-tag", default="full5",
                   help="Tag appended to MLflow run names and output CSV filename")
    p.add_argument("--use-oos", action="store_true",
                   help="Load v2_tuned signals from feat_alpha_ml_xgb_v2_tuned_oos (Path B)")
    args = p.parse_args()

    global FEATURES
    FEATURES = [f.strip() for f in args.features.split(",") if f.strip()]
    invalid = [f for f in FEATURES if f not in ALL_FEATURES]
    if invalid:
        raise ValueError(f"Unknown feature(s): {invalid}. Allowed: {ALL_FEATURES}")
    log.info("Feature set (%d): %s", len(FEATURES), FEATURES)
    log.info("Run tag: %s", args.run_tag)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.db), read_only=True)
    df = load_joined(con, use_oos=args.use_oos)
    con.close()

    # Median-impute per-feature (global)
    medians = {c: df[c].median() for c in FEATURES}
    for c in FEATURES:
        df[c] = df[c].fillna(medians[c])

    # Cross-sectional z-score per date, in-place on feature cols
    log.info("Cross-sectional z-scoring inputs per date...")
    df = cross_sectional_zscore(df, FEATURES)

    all_rows = []
    for ty in CV_TEST_YEARS:
        tr, te = purged_train_test_split(df, ty, PURGE_DAYS)
        if tr is None:
            log.warning("Skipping %d (insufficient train)", ty)
            continue
        X_tr, y_tr = tr[FEATURES], tr[TARGET].to_numpy()
        X_te, y_te = te[FEATURES], te[TARGET].to_numpy()

        # Inner validation: hold out last 10% of training dates for ridge alpha tuning
        train_dates = sorted(tr["date"].unique())
        val_cut = train_dates[int(len(train_dates) * 0.9)]
        inner_tr_mask = tr["date"] <= val_cut
        X_inner_tr = tr.loc[inner_tr_mask, FEATURES]
        y_inner_tr = tr.loc[inner_tr_mask, TARGET].to_numpy()
        X_inner_val = tr.loc[~inner_tr_mask, FEATURES]
        y_inner_val = tr.loc[~inner_tr_mask, TARGET].to_numpy()

        best_sig_name, best_sig_ic = best_single_signal_ic(te)
        per_fold = {
            "test_year": ty, "n_train": len(tr), "n_test": len(te),
            "best_single_signal": best_sig_name,
            "best_single_ic": best_sig_ic,
        }

        for variant in ("equal_weight", "ridge", "xgb_thin"):
            t0 = time.time()
            if variant == "ridge":
                # Re-fit on full X_tr using tuned alpha on inner split
                pred = fit_and_predict(
                    variant, X_inner_tr, y_inner_tr, X_te,
                    ridge_inner_val=(X_inner_val, y_inner_val),
                )
                # Also record on full training set with tuned alpha for final model
            else:
                pred = fit_and_predict(variant, X_tr, y_tr, X_te)
            res = evaluate(variant, pred, te)
            res["fit_time_s"] = round(time.time() - t0, 2)
            res["test_year"] = ty
            res["best_single_signal"] = best_sig_name
            res["best_single_ic"] = best_sig_ic
            res["ic_lift_vs_best_single"] = res["pooled_ic"] - best_sig_ic
            res["ic_lift_pct"] = (
                (res["pooled_ic"] - best_sig_ic) / abs(best_sig_ic) * 100
                if best_sig_ic and not np.isnan(best_sig_ic) else float("nan")
            )
            log.info(
                "  %d/%s  pooled_ic=%+0.4f  pd_ic=%+0.4f  q_spread=%+0.1fbps  "
                "best_single=%s(%+0.4f)  lift=%+0.2f%%",
                ty, variant,
                res["pooled_ic"], res["per_date_ic_mean"],
                res["quintile_spread_bps"],
                best_sig_name, best_sig_ic, res["ic_lift_pct"],
            )
            all_rows.append(res)

    rdf = pd.DataFrame(all_rows)
    out_csv = args.output_dir / f"cv_results_ensemble_{args.run_tag}.csv"
    rdf.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    log.info("\n" + "=" * 70)
    log.info("CV SUMMARY")
    log.info("=" * 70)
    for variant in ("equal_weight", "ridge", "xgb_thin"):
        sub = rdf[rdf["variant"] == variant]
        log.info(
            "%s  mean pooled_ic=%+0.4f  std=%0.4f  mean pd_ic=%+0.4f  "
            "mean q_spread=%+0.1fbps  mean lift=%+0.2f%%",
            variant,
            sub["pooled_ic"].mean(), sub["pooled_ic"].std(),
            sub["per_date_ic_mean"].mean(),
            sub["quintile_spread_bps"].mean(),
            sub["ic_lift_pct"].mean(),
        )

    # Best single signal baseline (year-by-year and mean)
    best_row = (rdf.groupby("test_year")[["best_single_signal", "best_single_ic"]]
                .first().reset_index())
    log.info("\nBest single signal per year:")
    for _, r in best_row.iterrows():
        log.info("  %d  %-30s  %+0.4f", int(r["test_year"]),
                 r["best_single_signal"], r["best_single_ic"])
    log.info("  mean best_single_ic=%+0.4f", best_row["best_single_ic"].mean())

    # MLflow
    if args.mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("ensemble_stacker")
            for variant in ("equal_weight", "ridge", "xgb_thin"):
                sub = rdf[rdf["variant"] == variant]
                with mlflow.start_run(run_name=f"path_A_{args.run_tag}_{variant}"):
                    mlflow.log_params({"variant": variant, "test_years": CV_TEST_YEARS,
                                       "features": FEATURES})
                    mlflow.log_metrics({
                        "mean_pooled_ic": sub["pooled_ic"].mean(),
                        "mean_per_date_ic": sub["per_date_ic_mean"].mean(),
                        "mean_q_spread_bps": sub["quintile_spread_bps"].mean(),
                        "mean_ic_lift_pct": sub["ic_lift_pct"].mean(),
                    })
                    mlflow.log_artifact(str(out_csv))
            log.info("MLflow runs logged.")
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
