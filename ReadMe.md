# 🧠 Kairos: Complete Universe Modeling Pipeline (v2)

This pipeline builds **stable, reproducible, and research-grade predictive models** on a fixed universe of US mid/large-cap equities. It's structured for **modularity, forward compatibility**, and **iterative alpha discovery**.

---

## 📐 Project Goals

- Build a **time-consistent universe** of tradable tickers  
- Generate a **historical feature matrix** with scalable alpha features  
- Train **modular, swappable models** on multi-horizon targets  
- Simulate **top-k strategies** to test tradeability  
- Log results to **DuckDB** for meta-analysis  
- Enable **iterative refinement and ensemble stacking**  

## 🛠️ Pipeline Overview
```flow
        ┌────────────────────────────┐
        │   sep_base / mid_cap_xx    │
        └────────────┬───────────────┘
                     ▼
┌────────────────────────────────────────────┐
│ filter/filter_universe_by_start_date.py    │
│ ➤ Outputs: midcap_<year>universe           │
└────────────────┬───────────────────────────┘
                 ▼
┌────────────────────────────────────────────┐
│ refresh_feature_tables.py                  │
│ ➤ Runs all features/feat.py scripts        │
│ ➤ Outputs: feat_price_action, feat_trend…  │
└────────────────┬───────────────────────────┘
                 ▼
┌────────────────────────────────────────────┐
│ features/build_feat_matrix_complete.py     │
│ ➤ --full joins all feat_ tables            │
│ ➤ Outputs: feat_matrix_complete_<year>     │
└────────────────┬───────────────────────────┘
                 ▼
┌────────────────────────────────────────────┐
│ scripts/generate_targets.py                │
│ ➤ Computes: ret_1d_f, ret_5d_f, ret_21d_f  │
│ ➤ Outputs: feat_matrix_targets_<year>      │
└────────────────┬───────────────────────────┘
                 ▼
┌────────────────────────────────────────────┐
│ models/train_model.py                      │
│ ➤ Trains Ridge / LGBM on target matrix     │
│ ➤ Saves: model.pkl, predictions.csv        │
│ ➤ Logs SHAP (LGBM only)                    │
└────────────────┬───────────────────────────┘
                 ▼
┌────────────────────────────────────────────┐
│ scripts/simulate_strategy.py               │
│ ➤ Simulates top-K strategy from predictions│
│ ➤ Logs to DuckDB: strategy_stats           │
└────────────────────────────────────────────┘
```

## 🚀 Step-by-Step Usage

### 0. Daily Downloads ( Perform Every Trading Day )

```python
python scripts/daily_download.py --date 2025-07-25
python scripts/merge_daily_download_duck.py --update-golden '/mnt/ssd_quant/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb'  
```
### 1. Filter Modeling Universe by Start Date

```python
python filter/filter_universe_by_start_date.py --year 2008
python filter/filter_universe_by_start_date.py --year 2014
```
### 2. Generate All Feature Tables
   ```python
   python refresh_feature_tables.py
   ```
### 3. Build Feature Matrix
#### Base version (3 features)
```python
python features/build_feat_matrix_complete.py --year 2008
```
#### Full version (joins all feat\_\* tables)
```python
python features/build_feat_matrix_complete.py --year 2008 --full
```
### 4. Add Forward Targets
```python
python scripts/generate_targets.py --year 2008 5. Train a Model
```
### 5. Train Model
#### Ridge
```python
python models/train_model.py --config models/config/ridge_2008.yaml --year 2008
```
#### LightGBM
```python
python models/train*model.py --config models/config/lgbm_2008.yaml --year 2008 
```
### 6. Simulate Strategy and Log to DuckDB
```python
python scripts/simulate_strategy.py --pred-file models/output/predictions_ridge_2008.csv --tag ridge_v1
python scripts/simulate_strategy.py --pred-file models/output/predictions_lgbm_2008.csv --tag lgbm_final
```
# ✅ After Running
Output Description
midcap_2008_universe Filtered tickers with coverage
feat_matrix_complete_2008 Feature matrix with engineered features
feat_matrix_targets_2008 Matrix + future return targets
\*.pkl Trained model file
predictions*_.csv Model predictions + actuals
metrics\__.csv MSE, R², Sharpe, accuracy
shap\_\*.csv Feature importance (LGBM only)
strategy_stats (DuckDB) Logged results: Sharpe, drawdown, etc.

# 🧪 Example: Full 2008 Run
```python
python scripts/daily_download.py --date 2025-07-25  
python scripts/merge_daily_download_duck.py --update-golden '/mnt/ssd_quant/media/vjerome2/Extreme Pro/kairos_phase4/data/kairos.duckdb'
python filter/filter_universe_by_start_date.py --year 2008
python refresh_feature_tables.py
python features/build_feat_matrix_complete.py --year 2008 --full
python scripts/generate_targets.py --year 2008
python models/train_model.py --config models/config/ridge_2008.yaml --year 2008
python models/train_model.py --config models/config/lgbm_2008.yaml --year 2008
python scripts/simulate_strategy.py --pred-file models/output/predictions_ridge_2008.csv --tag ridge_v1
python scripts/simulate_strategy.py --pred-file models/output/predictions_lgbm_2008.csv --tag lgbm_final
python scripts/simulate_strategy.py --pred-file models/output/predictions_mlp_2008.csv --tag mlp_v1
python scripts/simulate_strategy.py --pred-file models/output/predictions_xgb_2008.csv --tag xgb_v1
```
