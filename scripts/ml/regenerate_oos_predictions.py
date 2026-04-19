#!/usr/bin/env python3
"""
regenerate_oos_predictions.py
=============================
Walk-forward regenerate v2_tuned predictions so each test-year's predictions
come from a model that did NOT see that year in training. Stitches the
per-fold OOS predictions into a single table `feat_alpha_ml_xgb_v2_tuned_oos`
for use by the ensemble stacker (Path B — clean OOS).

For years < first test year (2019), no OOS prediction is available; those rows
are omitted from the output table. The ensemble stacker will naturally filter
to test years 2019-2025.

Uses GPU XGBoost (device=cuda, tree_method=hist) for speed.

Usage:
  python scripts/ml/regenerate_oos_predictions.py --db data/kairos.duckdb
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

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("regenerate_oos")

# Mirror exactly what v2_tuned uses — copy/pasted from train_xgb_alpha_v2_tuned.py
FEATURE_SOURCES = {
    "feat_fundamental":   ["earnings_yield","fcf_yield","roa","book_to_market",
                           "operating_margin","roe"],
    "feat_vol_sizing":    ["vol_21","vol_63","vol_blend"],
    "feat_beta":          ["beta_21d","beta_63d","beta_252d","resid_vol_63d"],
    "feat_price_action":  ["hl_ratio","range_pct","ret_21d","ret_5d"],
    "feat_momentum_v2":   ["mom_1m","mom_3m","mom_6m","mom_12m","mom_12_1","reversal_1m"],
}
FEATURES = [c for cols in FEATURE_SOURCES.values() for c in cols]

TARGET_REG = "ret_5d_f"
TARGET_CLF = "label_5d_up"

CV_TEST_YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
PURGE_DAYS = 5

XGB_COMMON = dict(
    learning_rate=0.0996, max_depth=4, subsample=0.545,
    colsample_bytree=0.857, reg_lambda=0.50, reg_alpha=0.644,
    min_child_weight=408, tree_method="hist", random_state=42,
    n_jobs=-1,
)
XGB_REG = dict(objective="reg:squarederror", **XGB_COMMON)
XGB_CLF = dict(objective="binary:logistic", eval_metric="auc", **XGB_COMMON)
N_ESTIMATORS = 500
EARLY_STOPPING_ROUNDS = 50


def load_data(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    log.info("Loading features + targets...")
    q = """
        SELECT ticker, date, ret_5d_f
        FROM feat_targets
        WHERE date >= '2015-01-01' AND ret_5d_f IS NOT NULL
    """
    df = con.execute(q).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    log.info("  base grid: %d rows", len(df))
    for table, cols in FEATURE_SOURCES.items():
        log.info("  joining %s...", table)
        cols_s = ", ".join(cols)
        f = con.execute(f"SELECT ticker, date, {cols_s} FROM {table}").fetchdf()
        f["date"] = pd.to_datetime(f["date"])
        df = df.merge(f, on=["ticker", "date"], how="left")
    df[TARGET_CLF] = (df[TARGET_REG] > 0).astype(int)
    df["year"] = df["date"].dt.year
    log.info("  final: %d rows, %d features", len(df), len(FEATURES))
    return df


def split_with_purge(df: pd.DataFrame, test_year: int, purge_days: int):
    test = df[df["year"] == test_year]
    train = df[df["year"] < test_year]
    if train.empty or test.empty:
        return None, None
    tdates = sorted(train["date"].unique())
    if len(tdates) <= purge_days:
        return None, None
    cut = tdates[-(purge_days + 1)]
    train = train[train["date"] <= cut].copy()
    test = test.copy()
    return train, test


def train_one_fold(train_df, test_df, features, use_gpu=True):
    X_tr = train_df[features].copy()
    X_te = test_df[features].copy()
    y_tr_reg = train_df[TARGET_REG].values
    y_te_reg = test_df[TARGET_REG].values
    y_tr_clf = train_df[TARGET_CLF].values
    y_te_clf = test_df[TARGET_CLF].values

    # Median impute from train
    medians = {c: X_tr[c].median() for c in features}
    for c in features:
        X_tr[c] = X_tr[c].fillna(medians[c])
        X_te[c] = X_te[c].fillna(medians[c])

    # Inner validation split (last 10% of train by date)
    n_val = int(len(X_tr) * 0.1)
    X_tr_fit = X_tr.iloc[:-n_val]; X_val = X_tr.iloc[-n_val:]
    y_tr_fit_reg = y_tr_reg[:-n_val]; y_val_reg = y_tr_reg[-n_val:]
    y_tr_fit_clf = y_tr_clf[:-n_val]; y_val_clf = y_tr_clf[-n_val:]

    extra = {"device": "cuda"} if use_gpu else {}

    reg = xgb.XGBRegressor(n_estimators=N_ESTIMATORS,
                           early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                           **XGB_REG, **extra)
    reg.fit(X_tr_fit, y_tr_fit_reg, eval_set=[(X_val, y_val_reg)], verbose=False)
    reg_pred_test = reg.predict(X_te)

    clf = xgb.XGBClassifier(n_estimators=N_ESTIMATORS,
                            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            **XGB_CLF, **extra)
    clf.fit(X_tr_fit, y_tr_fit_clf, eval_set=[(X_val, y_val_clf)], verbose=False)
    clf_pred_test = clf.predict_proba(X_te)[:, 1]

    from scipy.stats import spearmanr
    ic_reg, _ = spearmanr(reg_pred_test, y_te_reg)
    ic_clf, _ = spearmanr(clf_pred_test, y_te_reg)
    return reg_pred_test, clf_pred_test, float(ic_reg), float(ic_clf), \
           reg.best_iteration, clf.best_iteration


def main() -> int:
    p = argparse.ArgumentParser(description="Walk-forward regenerate v2_tuned OOS predictions")
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--output-table", default="feat_alpha_ml_xgb_v2_tuned_oos")
    p.add_argument("--cpu", action="store_true", help="Force CPU XGBoost")
    p.add_argument("--years", default=",".join(str(y) for y in CV_TEST_YEARS),
                   help="Comma-separated test years")
    args = p.parse_args()

    years = [int(y) for y in args.years.split(",")]
    use_gpu = not args.cpu

    con = duckdb.connect(str(args.db), read_only=False)
    df = load_data(con)

    all_preds = []
    ic_summary = []
    t_total = time.time()
    for test_year in years:
        tr, te = split_with_purge(df, test_year, PURGE_DAYS)
        if tr is None:
            log.warning("Skipping %d — no train data", test_year)
            continue
        log.info("Fold %d: train=%d  test=%d", test_year, len(tr), len(te))
        t0 = time.time()
        reg_p, clf_p, ic_r, ic_c, nt_r, nt_c = train_one_fold(
            tr, te, FEATURES, use_gpu=use_gpu,
        )
        dt = time.time() - t0
        log.info("  fit %.1fs  trees reg=%d clf=%d  IC reg=%+.4f  IC clf=%+.4f",
                 dt, nt_r, nt_c, ic_r, ic_c)
        preds = te[["ticker", "date"]].copy()
        preds["alpha_ml_v2_tuned_reg_oos"] = reg_p.astype(np.float32)
        preds["alpha_ml_v2_tuned_clf_oos"] = clf_p.astype(np.float32)
        all_preds.append(preds)
        ic_summary.append({"test_year": test_year, "ic_reg": ic_r, "ic_clf": ic_c,
                           "n_train": len(tr), "n_test": len(te),
                           "n_trees_reg": nt_r, "n_trees_clf": nt_c,
                           "fit_time_s": round(dt, 1)})

    out = pd.concat(all_preds, ignore_index=True)
    log.info("Total OOS predictions: %d rows across %d years (%.1fs total)",
             len(out), len(ic_summary), time.time() - t_total)

    log.info("Writing to %s...", args.output_table)
    con.execute(f"DROP TABLE IF EXISTS {args.output_table}")
    con.register("preds_df", out)
    con.execute(f"""
        CREATE TABLE {args.output_table} AS
        SELECT ticker, CAST(date AS DATE) AS date,
               alpha_ml_v2_tuned_reg_oos,
               alpha_ml_v2_tuned_clf_oos
        FROM preds_df
    """)
    n = con.execute(f"SELECT COUNT(*) FROM {args.output_table}").fetchone()[0]
    log.info("  wrote %d rows to %s", n, args.output_table)

    summary_df = pd.DataFrame(ic_summary)
    out_csv = Path("scripts/ml/outputs") / "cv_results_v2_tuned_oos_regen.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False)
    log.info("Wrote fold summary to %s", out_csv)
    log.info("\n%s", summary_df.to_string(index=False))
    log.info("\nMean IC reg: %+.4f   Mean IC clf: %+.4f",
             summary_df["ic_reg"].mean(), summary_df["ic_clf"].mean())

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
