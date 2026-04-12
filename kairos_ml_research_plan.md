# Kairos ML Research Plan
## Phase 4 — Equity Alpha Research
*Last updated: April 12, 2026*

---

## System Context (include in every new chat)

```
Project: Kairos Phase 4 — Equity Alpha Research
Language: Python 3.11
Database: DuckDB 1.4.3
DB Path: /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb

Container: kairos-ml-lab (Docker)
  - JupyterLab: http://localhost:8888/lab
  - MLflow:     http://localhost:5000
  - Models out: /media/.../Extreme Pro/ml_experiments/models/
  - MLflow DB:  /media/.../Extreme Pro/ml_experiments/mlflow/

Current production model: scripts/ml/train_xgb_alpha_v2_tuned.py
  - 23 features from: feat_fundamental, feat_vol_sizing, feat_beta,
    feat_price_action, feat_momentum_v2
  - Target: ret_5d_f (raw 5-day forward return)
  - Mean IC: ~0.04 (4%)
  - In production: alpha_ml_v2_tuned_clf drives weekly rebalance of top 75 stocks

Key tables in DuckDB:
  feat_targets          — ticker, date, ret_5d_f, label_5d_up
  feat_matrix_v2        — all joined features + alpha signals
  sep_base_academic     — daily prices for universe (2,335 tickers)
  feat_momentum_v2      — momentum factors (19,643 tickers, filter to universe)
  feat_quality_v2       — ROE, ROA, accruals
  feat_value_v2         — earnings/book/EBITDA yields
  feat_fundamental      — forward-filled SF1 fundamentals
  feat_vol_sizing       — vol_21, vol_63, vol_blend
  feat_beta             — beta_21d, beta_63d, beta_252d, resid_vol_63d
  feat_price_action     — hl_ratio, range_pct, ret_21d, ret_5d
  feat_composite_v33_regime — production alpha signal
  tickers               — metadata (sector, exchange, category)
                          NOTE: has 2 rows per ticker (SF1 + SEP table rows)
                          Always join with: SELECT DISTINCT ticker, sector FROM tickers

DuckDB connection (always read-only from container):
  import duckdb
  con = duckdb.connect('/data/kairos.duckdb', read_only=True)

Write outputs to: /models/ and log to MLflow at http://localhost:5000
```

---

## Git Branch Strategy

```
main
  └── sector-neutral          ← current working branch
        └── fix/validation-regime   ← next branch (not yet created)
```

**main** — production pipeline, never broken
**sector-neutral** — contains 5 new scripts (see below), pushed and safe
**fix/validation-regime** — to be created, fixes validation regime pipeline-wide

---

## Scripts Added (sector-neutral branch)

| Script | Location | Status | Purpose |
|--------|----------|--------|---------|
| `train_xgb_alpha_v3_neutral.py` | `scripts/ml/` | ✓ Written, runs | Train on sector-neutral target |
| `generate_ml_predictions_v3_neutral.py` | `scripts/ml/` | ✓ Written, runs | Generate v3 predictions |
| `build_gross_profit.py` | `scripts/features/` | ✓ Pushed | Feature builder |
| `train_xgb_alpha_v3.py` | `scripts/ml/` | ✓ Pushed | Intermediate version |
| `tune_xgb_optuna_v3.py` | `scripts/ml/` | ✓ Pushed | Optuna tuning script |

**Note:** `generate_ml_predictions_v3_neutral.py` produces negative IC (-0.0065)
in validation due to the undertrained final model (6 trees). This is a known
issue caused by the validation regime flaw described below. Do NOT add to
pipeline until the validation regime fix is complete and verified.

---

## Known Issues

### Issue 1 — Validation Regime Flaw (CRITICAL, pipeline-wide)

**What it is:**
All training scripts use the last 10% of training data chronologically
as the early stopping validation set:

```python
n_val = int(n_train * 0.1)
X_val = X_train.iloc[-n_val:]
```

**Why it's wrong:**
The last 10% of training is always a specific market regime (e.g., late 2022
bear market when training for 2023). Early stopping sees no improvement on that
regime and stops prematurely. This produces undertrained models with very low
tree counts (0-6 trees in some folds).

**Evidence:**
- CV clf tree counts for v3_neutral: 6, 0, 270, 41, 0, 169, 25
- Final model clf stopped at 6 trees → std of predictions = 0.0081 (nearly flat)
- Validation IC = -0.0065 vs CV IC = +0.0281 (catastrophic collapse)

**Affected scripts:**
- `train_xgb_alpha_v2_tuned.py` (production — masked by noisier target)
- `train_xgb_alpha_v3_neutral.py` (new — exposed by cleaner target)
- All future training scripts

**The fix:**
Use a dedicated validation year between training and test, not a slice of
training data. When testing year N, train on years < N-1, validate on year N-1,
test on year N. Early stopping sees a full representative year.

```python
# Current (wrong)
train = data[data.year < test_year]
val   = train.iloc[-10%:]  # last slice of training

# Fixed (correct)
train = data[data.year < test_year - 1]
val   = data[data.year == test_year - 1]  # dedicated validation year
test  = data[data.year == test_year]
```

**Status:** Not yet implemented. Must be proven in research notebook before
touching any production scripts.

---

### Issue 2 — tickers Table Duplicate Rows (KNOWN, handled)

**What it is:**
The `tickers` table has 2 rows per ticker — one for SF1, one for SEP.
Both rows have identical sector values.

**Fix:**
Always join with:
```sql
LEFT JOIN (
    SELECT DISTINCT ticker, sector
    FROM tickers
    WHERE sector IS NOT NULL
) tk ON t.ticker = tk.ticker
```

**Status:** Fixed in all new scripts. Must be applied to any new scripts.

---

### Issue 3 — feat_momentum_v2 Universe Mismatch (KNOWN, handled)

**What it is:**
`feat_momentum_v2` contains 19,643 tickers (full Sharadar universe) vs
2,335 tickers in the active universe. When joining, always filter to
tickers in `feat_targets` or `sep_base_academic`.

**Status:** Handled automatically by LEFT JOIN in training scripts.
Explicitly filtered in research DB creation (see below).

---

### Issue 4 — GNLN Beta Outlier (KNOWN, handled in research)

**What it is:**
Ticker GNLN has beta_252d values up to 906 due to extreme price event.
Only 0.15% of rows have |beta| > 3.

**Fix in research notebooks:**
Winsorize beta at ±3 before use in neutralization calculations.

**Status:** Not a production issue (beta is used as a feature, not in
neutralization in production scripts). Documented for awareness.

---

## Path 1 — Target Neutralization

### Research Findings (April 12, 2026)

**Notebook:** `path1_neutral_target.ipynb` in JupyterLab

**MLflow experiment:** `path1_target_neutralization` (experiment ID: 2)

**Walk-forward CV results (2019-2025):**

| Variant | Mean IC | Std IC | IC Sharpe | % Positive |
|---------|---------|--------|-----------|------------|
| baseline_raw | 0.0244 | 0.0340 | 0.717 | 86% |
| market_neutral | 0.0242 | 0.0215 | 1.125 | 86% |
| **sector_neutral** | **0.0281** | **0.0176** | **1.598** | **100%** |
| full_neutral_v2 | 0.0233 | 0.0233 | 1.001 | 86% |

**Winner: sector_neutral**
- +15% mean IC improvement over baseline
- +106% IC Sharpe improvement
- 100% positive folds (never a losing year in test period)

**Sector neutral formula:**
```python
ret_5d_sector_neutral = ret_5d_f - equal_weight_sector_mean_ret_5d
```

No beta involved. No market neutralization. Just remove what the sector did.

**Why sector neutral won over full neutral:**
Market effects in this universe are already partially captured by beta and
momentum features. Sector rotation was the true unpredictable noise.

**IC by year (sector_neutral):**
```
2019: 0.0329
2020: 0.0222
2021: 0.0463
2022: 0.0306
2023: 0.0030  ← weakest year
2024: 0.0548  ← strongest year
2025: 0.0072
```

### Current Blocker

Final model is undertrained (6 trees) due to validation regime flaw (Issue 1).
Prediction script produces IC = -0.0065 despite CV showing IC = +0.0281.

**Do not add to pipeline until validation regime fix is complete.**

### Next Steps for Path 1

1. Fix validation regime in research environment
2. Prove fix works in notebook (MLflow comparison)
3. Retrain v3_neutral with fixed regime
4. Verify prediction script produces positive IC
5. Add to pipeline Phase 6
6. Apply same fix to v2_tuned

---

## Research Environment Plan

### Purpose
Fixed snapshot of production data for safe ML experimentation.
Never updated. Never affects production database.

### Location
```
/media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos_research.duckdb
```

### Tables to Copy

| Table | Filter | Approx Rows |
|-------|--------|-------------|
| `feat_targets` | 2015-2025 | 5,288,553 |
| `feat_fundamental` | 2015-2025 | 5,288,553 |
| `feat_vol_sizing` | 2015-2025 | 5,235,032 |
| `feat_beta` | 2015-2025 | 5,235,032 |
| `feat_price_action` | 2015-2025 | 5,270,821 |
| `feat_momentum_v2` | 2015-2025 + universe tickers only | ~5,000,000 |
| `sep_base_academic` | 2015-2025 | 5,288,553 |
| `tickers` | all rows | 60,277 |

**Total: ~41M rows, estimated <10GB on SSD**

### Script to Create
`scripts/research/create_research_db.py`
- Reads from production DB (read-only)
- Writes to research DB (new file)
- Never modifies production DB
- Run once, then research DB is frozen

### Status
Not yet created. Next immediate step.

---

## Remaining Research Paths

### Path 2 — CPCV (Combinatorial Purged Cross-Validation)

**What:** Replace walk-forward CV with CPCV to get honest IC estimates
and catch overfitting to specific train/test boundaries.

**Why:** Walk-forward CV with 7 folds gives 7 IC estimates. CPCV with
C(6,2)=15 combinations gives a distribution. IC Sharpe (mean/std of
distribution) becomes the model selection criterion.

**Dependencies:** Validation regime fix must be done first.

**Success criteria:**
- CPCV mean IC > 0.02
- CPCV IC Sharpe > 1.0
- % folds with IC > 0 exceeds 70%

**Status:** Not started.

---

### Path 3 — Model Ensemble (XGBoost + LightGBM + CatBoost)

**What:** Train 3 diverse models, combine predictions.

**Why:** Different tree-building algorithms make partially uncorrelated
errors. Averaging reduces prediction variance.

**Key requirement:** Pairwise prediction correlation < 0.90 for ensemble
to add value. Introduce feature diversity (random 80% subsets per model).

**Success criteria:** Ensemble IC exceeds best individual model by 0.005.

**Status:** Not started.

---

### Path 4 — Cross-Sectional Rank Features

**What:** Transform raw features to percentile ranks within date/sector/size.

**Why:** Absolute values (momentum = +23%) are regime-dependent. Ranks
(87th percentile within sector) are invariant to market level.

**Variants to test:**
- Model A: 23 raw features (baseline)
- Model B: 23 raw + 23 global rank
- Model C: 23 raw + 23 global rank + 23 sector rank

**Success criteria:** Model B or C IC exceeds baseline by 0.003.

**Status:** Not started.

---

### Path 5 — Feature Orthogonalization

**What:** Remove correlations between features using PCA within feature groups.

**Why:** vol_21, vol_63, vol_blend are nearly identical. roe and roa are
both profitability. Correlated features cause unstable importances and
double-counting.

**Approach:** Partial orthogonalization within groups (volatility, beta,
profitability, momentum, value, price) — not full PCA which loses
interpretability.

**Success criteria:** IC improvement 0.002+, feature importance std
across folds decreases 20%+.

**Status:** Not started.

---

## Recommended Execution Order

| Priority | Path | Status | Blocker |
|----------|------|--------|---------|
| 0 | Fix validation regime | 🔴 In progress | None |
| 0 | Create research DB | 🔴 Not started | None |
| 1 | Path 1 — Target Neutralization | 🟡 Proven, blocked | Validation regime fix |
| 2 | Path 2 — CPCV | 🔴 Not started | Validation regime fix |
| 3 | Path 3 — Ensemble | 🔴 Not started | Paths 1+2 |
| 4 | Path 4 — Rank Features | 🔴 Not started | Paths 1+2 |
| 5 | Path 5 — Orthogonalization | 🔴 Not started | Paths 1+2 |

---

## MLflow Experiments

| Experiment | ID | Runs | Status |
|------------|-----|------|--------|
| test_connection | 1 | 0 | Ignore |
| path1_target_neutralization | 2 | 4 | Complete |

---

## Key Decisions Made

1. **Research in notebooks first, promote to pipeline only after proof.**
   Never experiment directly on production scripts.

2. **New scripts use versioned naming** (`_v3_neutral`) and never overwrite
   existing tables or model files.

3. **IC is always evaluated against raw `ret_5d_f`** regardless of what
   target the model was trained on. Raw return is what the market delivers.

4. **Classification model drives rebalance, not regression.** Regression
   output is saved for research but not used in production signal.

5. **tickers join always uses DISTINCT** to avoid duplicate rows from
   SF1/SEP table entries.

6. **feat_momentum_v2 always filtered to universe tickers** when used in
   training to avoid the 19,643 ticker full universe contaminating the
   2,335 ticker academic universe.

7. **Research DB is a frozen snapshot.** It is never updated. Proven
   improvements are promoted to production pipeline scripts only.

---

*Document maintained across chat sessions. Include system context block
at top of every new chat.*
