# Kairos ML Research Status
## Phase 4 — Equity Alpha Research
*Last updated: April 12, 2026 — End of session 2*

---

## System Context (include in every new chat)

```
Project: Kairos Phase 4 — Equity Alpha Research
Language: Python 3.11
Database: DuckDB 1.4.3

Production DB (read-only):
  /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb

Research DB (read-only, frozen snapshot):
  /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos_research.duckdb

Container: kairos-ml-lab (Docker)
  - JupyterLab: http://localhost:8888/lab
  - MLflow:     http://localhost:5000
  - Models out: /media/.../Extreme Pro/ml_experiments/models/
  - MLflow DB:  /media/.../Extreme Pro/ml_experiments/mlflow/
  - Notebooks:  /media/.../Extreme Pro/ml_experiments/notebooks/ → /workspace (bind mounted)

Inside container paths:
  - Research DB:  /data/kairos_research.duckdb  (read-only)
  - Production DB: /data/kairos.duckdb          (read-only)
  - Models:       /models/
  - Notebooks:    /workspace/

Current production model: scripts/ml/train_xgb_alpha_v2_tuned.py
  - 23 features from: feat_fundamental, feat_vol_sizing, feat_beta,
    feat_price_action, feat_momentum_v2
  - Target: ret_5d_f (raw 5-day forward return)
  - Mean IC: ~0.04 (4%)
  - In production: alpha_ml_v2_tuned_clf drives weekly rebalance of top 75 stocks

Proven next model (not yet in production):
  - File: /models/xgb_v3_neutral_n100_20260412.joblib
  - Features: /models/xgb_v3_neutral_n100_20260412_features.txt
  - Target: ret_5d_sector_neutral
  - n_estimators: 100, no early stopping
  - CPCV mean IC: 0.0259, IC Sharpe: 3.727, 100% positive folds

Key tables in research DB:
  feat_targets          — ticker, date, ret_5d_f, label_5d_up (5,288,553 rows, 2335 tickers, 2015-2025)
  feat_fundamental      — forward-filled SF1 fundamentals
  feat_vol_sizing       — vol_21, vol_63, vol_blend
  feat_beta             — beta_21d, beta_63d, beta_252d, resid_vol_63d
  feat_price_action     — hl_ratio, range_pct, ret_21d, ret_5d
  feat_momentum_v2      — mom_1m, mom_3m, mom_6m, mom_12m, mom_12_1, reversal_1m
  sep_base_academic     — daily prices for universe
  tickers               — metadata (sector, exchange, category)

Research DB boundary: ret_5d_f is clean through 2025-12-12 only.
  Dates from 2025-12-15 onward have corrupted forward returns (std 0.29+,
  max 12-32x) due to the 5-day window running past the end of the snapshot.
  Never use ret_5d_f for evaluation after 2025-12-12 in research DB.

Standard tickers join (always use this exact form):
  LEFT JOIN (
      SELECT DISTINCT ticker, sector
      FROM tickers
      WHERE sector IS NOT NULL
        AND ticker != 'N/A'
  ) tk ON t.ticker = tk.ticker

DuckDB connection (always read-only):
  import duckdb
  con = duckdb.connect('/data/kairos_research.duckdb', read_only=True)
```

---

## Docker Container Configuration

Container rebuilt April 12, 2026 with /workspace bind mounted.
Full docker run command:

```bash
docker run -d \
  --name kairos-ml-lab \
  --gpus all \
  -p 8888:8888 \
  -p 5000:5000 \
  -v "/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb:/data/kairos.duckdb:ro" \
  -v "/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos_research.duckdb:/data/kairos_research.duckdb:ro" \
  -v "/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/ml_experiments/mlflow:/mlflow" \
  -v "/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/ml_experiments/models:/models" \
  -v "/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/ml_experiments/notebooks:/workspace" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  kairos-ml-lab
```

---

## Git Branch Strategy

```
main
  └── sector-neutral          ← current working branch
        └── fix/validation-regime   ← to be created after research proves fix
```

---

## Completed — Session 1 (April 12, 2026)

### 1. N/A Ticker Warning — RESOLVED
The tickers table contained the literal string 'N/A' as a ticker for
institutional 13F filers (not stocks). Fix: add AND ticker != 'N/A'
to the standard tickers join. Applied to all new notebooks and scripts.

### 2. Validation Regime Flaw — DIAGNOSED AND FIXED
Early stopping is fundamentally incompatible with financial time series.
Fix: remove early_stopping_rounds entirely. Fix n_estimators via CPCV.
Evidence: fixed regime produced +50% mean IC, variance halved, all folds positive.
MLflow: experiment validation_regime_comparison (ID: 3)

### 3. CPCV n_estimators Grid Search — COMPLETE
Decision: n_estimators=100, no early stopping.
Rationale: IC Sharpe 3.793, turnover 28.9%, all 15 folds positive.
MLflow: experiment cpcv_n_estimators_grid (ID: 4)

---

## Completed — Session 2 (April 12, 2026)

### 4. Path 1 — Target Neutralization v2 — COMPLETE

**Notebook:** `path1_target_neutralization_v2.ipynb`
**MLflow experiment:** `path1_target_neutralization_v2` (ID: 5), 4 runs

**Walk-forward CV results (sector-neutral target, n=100, no early stopping):**

| Year | IC_raw |
|------|--------|
| 2019 | +0.0401 |
| 2020 | -0.0205 |
| 2021 | +0.0372 |
| 2022 | +0.0486 |
| 2023 | +0.0312 |
| 2024 | +0.0447 |
| 2025 | +0.0348 |

| Metric | Value |
|--------|-------|
| Mean IC_raw | 0.0309 |
| Std IC_raw | 0.0234 |
| IC Sharpe | 1.320 |
| Min IC_raw | -0.0205 (2020) |
| % positive | 86% |

2020 negative IC fully explained by two unpredictable macro events:
- February: initial COVID selloff (Feb 19 regime break)
- November: Pfizer vaccine announcement (Nov 9 factor rotation)
Q2 and Q3 2020 were solidly positive. Not a model flaw.

**CPCV results (sector-neutral target, n=100, no early stopping):**

| Metric | Value |
|--------|-------|
| Mean IC_raw | 0.0259 |
| Std IC_raw | 0.0069 |
| IC Sharpe | 3.727 |
| Min IC_raw | 0.0174 (fold 6) |
| % positive | 100% |

All 15 folds positive. Genuine cross-regime signal confirmed.

**Final model:**
- File: `/models/xgb_v3_neutral_n100_20260412.joblib` (171.3 KB)
- Features: `/models/xgb_v3_neutral_n100_20260412_features.txt`
- pred_std: 0.0257 (healthy — confirms model is not degenerate)
- pred_mean: 0.5119 (correctly centered)

**Inference simulation (2025-12-12 — last clean date in research DB):**
- Market context: down week, Fed hawkish surprise Dec 18
- Top 75 realized: -0.82% vs universe -1.05% (outperformed by 23 bps)
- Bottom 75 realized: -2.84%
- Long-short spread: +2.02%
- Single-date IC_raw: 0.1230

Model is working correctly. Ready for production prediction script fix.

---

## Locked Production Parameters

```python
XGB_PARAMS = {
    'n_estimators':     100,       # FIXED — CPCV grid search
    'max_depth':        4,
    'learning_rate':    0.05,
    'subsample':        0.7,
    'colsample_bytree': 0.7,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'min_child_weight': 50,
    'objective':        'binary:logistic',
    'device':           'cuda',
    'random_state':     42,
    'verbosity':        0,
    # early_stopping_rounds: REMOVED PERMANENTLY
}

CPCV_CONFIG = {
    'n_splits':      6,
    'n_test_splits': 2,   # C(6,2) = 15 folds
    'purge_days':    7,   # per test group boundary
}
```

---

## Sector-Neutral Target — Confirmed Formula

```python
sectors = con.execute("""
    SELECT DISTINCT ticker, sector
    FROM tickers
    WHERE sector IS NOT NULL
      AND ticker != 'N/A'
""").fetchdf()

df = df.merge(sectors, on='ticker', how='left')

sector_mean = (
    df.groupby(['date', 'sector'])['ret_5d_f']
    .mean()
    .rename('sector_ret_5d')
    .reset_index()
)
df = df.merge(sector_mean, on=['date', 'sector'], how='left')
df['ret_5d_sector_neutral'] = df['ret_5d_f'] - df['sector_ret_5d']

# Confirmed properties:
# mean = 0.000000
# std  = 0.508200 (vs 0.511722 raw)
# missing = 0
```

IC always evaluated against raw ret_5d_f regardless of training target.

---

## Key Decisions

1. Early stopping removed permanently.
2. n_estimators=100 fixed via CPCV grid search.
3. CPCV is the standard CV methodology. IC Sharpe (mean/std) is the
   model selection criterion.
4. IC always evaluated against raw ret_5d_f.
5. tickers join always uses DISTINCT + AND ticker != 'N/A'.
6. Notebooks persist via /workspace bind mount.
7. Research DB ret_5d_f is clean through 2025-12-12 only. Any
   evaluation using forward returns must use dates on or before
   that date. Live forward returns require production DB.

---

## MLflow Experiments

| Experiment | ID | Runs | Status |
|---|---|---|---|
| Default | 0 | 0 | Ignore |
| test_connection | 1 | 0 | Ignore |
| path1_target_neutralization | 2 | 4 | Complete (session 1) |
| validation_regime_comparison | 3 | 3 | Complete (session 1) |
| cpcv_n_estimators_grid | 4 | 8 | Complete (session 1) |
| path1_target_neutralization_v2 | 5 | 4 | Complete (session 2) |

---

## Notebooks in /workspace

| Notebook | Status | Purpose |
|---|---|---|
| `fix_validation_regime.ipynb` | ✓ Complete | Diagnosed and fixed validation regime flaw |
| `cpcv_n_estimators_grid.ipynb` | ✓ Complete | CPCV grid search, locked n=100 |
| `path1_target_neutralization_v2.ipynb` | ✓ Complete | Proven sector-neutral target with fixed params |

---

## Immediate Next Step

Fix `generate_ml_predictions_v3_neutral.py` to use:
- Model: `/models/xgb_v3_neutral_n100_20260412.joblib`
- Features: `/models/xgb_v3_neutral_n100_20260412_features.txt`
- Live features from production DB
- Sector-neutral target (training only — inference uses raw features)
- Output: predictions for current live cross-section with realized
  return tracking capability

This script was previously broken (IC = -0.0065) due to the 6-tree
undertrained model. That blocker is now cleared.

---

## Research Paths Status

| Priority | Path | Status | Blocker |
|---|---|---|---|
| 0 | Fix validation regime | ✅ Complete | — |
| 0 | Create research DB | ✅ Complete | — |
| 0 | CPCV n_estimators grid | ✅ Complete | — |
| 1 | Path 1 — Target Neutralization v2 | ✅ Complete | — |
| Next | Fix prediction script | 🔴 Not started | None |
| 2 | Path 2 — CPCV as standard baseline | 🔴 Not started | Path 1 ✅ |
| 3 | Path 3 — Ensemble | 🔴 Not started | Paths 1+2 |
| 4 | Path 4 — Rank Features | 🔴 Not started | Paths 1+2 |
| 5 | Path 5 — Orthogonalization | 🔴 Not started | Paths 1+2 |

---

## Known Issues

### Issue 2 — tickers Table Duplicate Rows (KNOWN, handled)
Always join with DISTINCT + AND ticker != 'N/A'. Fixed in all new scripts.

### Issue 3 — feat_momentum_v2 Universe Mismatch (KNOWN, handled)
Contains 19,643 tickers vs 2,335 in active universe.
Handled automatically by LEFT JOIN to feat_targets.

### Issue 4 — GNLN Beta Outlier (KNOWN, handled in research)
beta_252d up to 906. Winsorize at ±3 in any notebook using beta
for neutralization calculations.

### Issue 5 — Research DB Forward Return Boundary (KNOWN)
ret_5d_f is corrupted from 2025-12-15 onward in research DB.
Last clean evaluation date: 2025-12-12.
Live forward returns require production DB.

---

*Document maintained across chat sessions.*
*Always include this full document at the start of every new chat.*