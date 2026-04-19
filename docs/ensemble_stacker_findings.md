# Ensemble Stacker — Findings (Negative Result)

**Status:** Not worth deploying. Does NOT add value under clean OOS discipline.
**Date:** 2026-04-19
**Branch:** `temporimutator`

---

## TL;DR

Built a walk-forward meta-stacker over 5 existing alpha signals (v2_tuned_clf,
v2_tuned_reg, v3_neutral_clf, alpha_composite_v8, alpha_composite_v33_regime).
Initial results (Path A, using production predictions as inputs) showed a ~3×
quintile-spread improvement — which turned out to be **entirely a leakage
artifact**.

Under true out-of-sample discipline (Path B, walk-forward regenerated base
predictions), Ridge and XGBoost meta-models **go negative on IC** in 5 of 6
test years. Equal-weight blending barely matches v2_tuned_clf alone.

**The ensemble stacker does not unlock additional alpha from the existing
signal set.** Documented here so we don't revisit.

---

## What was built

| File | Purpose |
|---|---|
| `scripts/ml/train_ensemble_stacker.py` | Walk-forward CV meta-stacker. Three variants: equal-weight, Ridge (alpha-tuned), thin XGBoost. Supports `--features` subset and `--use-oos` flag for Path B. |
| `scripts/ml/regenerate_oos_predictions.py` | Walk-forward regenerate v2_tuned predictions per test year. Writes to new table `feat_alpha_ml_xgb_v2_tuned_oos`. GPU XGBoost — full 7-fold sweep runs in ~2 min. |
| `scripts/ml/outputs/cv_results_ensemble_*.csv` | Per-fold CV results per run (full5, prune3, prune3_oos). |
| `scripts/ml/outputs/cv_results_v2_tuned_oos_regen.csv` | Fold summary from OOS regen. |

---

## Run ledger

### Path A (leaky baselines)

Using production predictions `feat_alpha_ml_xgb_v2_tuned` etc. directly.
Base models were trained on full history, so test-year inputs are
in-sample for the base model.

| Run | Features | Variant | Mean pooled IC | Mean Q-spread (bps) |
|---|---|---|---|---|
| full5 | all 5 | equal_weight | +0.0565 | +92.9 |
| full5 | all 5 | ridge | +0.0579 | +144.0 |
| full5 | all 5 | xgb_thin | +0.0575 | +145.8 |
| prune3 | v2_tuned_clf, v2_tuned_reg, v8 | equal_weight | +0.0572 | +122.9 |
| prune3 | v2_tuned_clf, v2_tuned_reg, v8 | **ridge** | **+0.0585** | **+147.0** |
| prune3 | v2_tuned_clf, v2_tuned_reg, v8 | xgb_thin | +0.0564 | +151.5 |
| — | — | **v2_tuned_clf alone (best single)** | **+0.0692** | +49.7 |

Looked great: ~3× Q-spread lift vs best single signal.

### Path B (clean OOS)

`regenerate_oos_predictions.py` walk-forward regenerates v2_tuned predictions
such that each test year's prediction comes from a model trained only on
prior years. Rule-based signals (v8, v33_regime) are always OOS by
construction.

OOS regen fold IC (matches v2_tuned's reported CV IC of 0.041 on clf):

| Year | IC reg | IC clf | n_trees_clf | fit_s |
|---|---|---|---|---|
| 2019 | −0.027 | +0.037 | 380 | 6.0 |
| 2020 | +0.046 | +0.020 | 481 | 7.4 |
| 2021 | −0.007 | +0.051 | 499 | 10.7 |
| 2022 | +0.003 | +0.050 | 495 | 13.7 |
| 2023 | +0.017 | +0.044 | 499 | 15.2 |
| 2024 | −0.000 | +0.039 | 490 | 17.7 |
| 2025 | +0.037 | +0.062 | 498 | 21.5 |
| **Mean** | **+0.010** | **+0.044** | | 108s total |

Ensemble results with these OOS inputs:

| Variant | Mean pooled IC | Mean per-date IC | Mean Q-spread (bps) |
|---|---|---|---|
| **Best single** (v8 or v2_tuned_clf_oos by year) | **+0.0295** | — | — |
| equal_weight | +0.0232 | +0.0283 | +70.2 |
| **ridge** | **−0.0060** | **−0.0038** | +80.3 |
| xgb_thin | −0.0052 | −0.0011 | +94.6 |

Ridge and XGBoost go negative in 5 of 6 test years. Equal-weight stays
positive but underperforms picking v2_tuned_clf alone.

### Path A vs Path B side-by-side

| Metric | Path A (leaky) | Path B (clean) | Gap |
|---|---|---|---|
| Best single signal IC | 0.069 | 0.030 | −56% |
| Ridge ensemble IC | 0.058 | **−0.006** | flipped |
| Ridge ensemble Q-spread | 147 bps | 80 bps | −46% |

The entire Path A "lift" was leakage.

---

## Why the ensemble fails under OOS discipline

1. **Base-signal distributions shift year-over-year.** Ridge coefficients
   learned on 2015-2020 are optimal for that era; they don't generalize to
   2021-2025 because the relationships between v2_tuned_clf, v2_tuned_reg,
   and v8 change as market regime shifts. Classic stacking failure under
   non-stationarity.

2. **Signals aren't independent enough.** Best orthogonal pair was
   v2_tuned_clf vs v2_tuned_reg (corr −0.14). Everything else is either
   highly correlated with v2_tuned (v3_neutral at 0.65) or a weak
   variant (v33_regime derived from v8).

3. **Meta-model variance exceeds base-model variance.** XGBoost meta has
   2× the per-year IC std of equal-weight (0.025 vs 0.011). Ridge is
   similarly unstable.

## What this rules out

- Meta-stacking over the existing 5 signals. Don't revisit.
- Fancy meta-architectures (neural stackers, LightGBM meta) — if Ridge
  and XGBoost both fail, nothing more expressive will succeed on the
  same inputs.

## What remains on the table

The negative result suggests the **signal set is too correlated** for
stacking to help. Two paths forward:

1. **Add truly orthogonal signals** (sentiment, short interest, analyst
   revisions, alternative data). Once we have 1-2 genuinely uncorrelated
   inputs, re-attempt stacking.
2. **Stop chasing alpha on this signal set; focus on risk overlays and
   portfolio construction.** Regime-aware cash sleeve, transaction costs,
   capacity-aware sizing — all of these generate wealth without needing
   new alpha.

Plan: proceed with (2) first (cheap, high-impact), then re-attempt
stacking only after (1) is in place.

---

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
mlflow ui --host localhost --port 5000 &

# Path A scoping (leaky baselines)
python scripts/ml/train_ensemble_stacker.py \
    --db data/kairos.duckdb \
    --features alpha_ml_v2_tuned_clf,alpha_ml_v2_tuned_reg,alpha_composite_v8 \
    --run-tag prune3

# Path B — regenerate clean OOS predictions (~2 min on GPU)
python scripts/ml/regenerate_oos_predictions.py --db data/kairos.duckdb

# Path B ensemble
python scripts/ml/train_ensemble_stacker.py \
    --db data/kairos.duckdb \
    --features alpha_ml_v2_tuned_clf,alpha_ml_v2_tuned_reg,alpha_composite_v8 \
    --run-tag prune3_oos --use-oos
```

Results stored in `scripts/ml/outputs/cv_results_ensemble_*.csv` and the
new DuckDB table `feat_alpha_ml_xgb_v2_tuned_oos` (3.66M rows, 2019-2025).
