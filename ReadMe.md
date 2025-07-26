🧠 Kairos: Complete Universe Modeling Pipeline (v1)

This branch implements a clean, consistent pipeline for building predictive models on a fixed universe of US mid/large-cap equities. It is designed for stability, forward compatibility, and iterative feature/model evolution.
📐 Project Goals

    Build a time-consistent universe of tradable tickers

    Generate a complete historical feature matrix per universe

    Train modular, swappable models on multiple return targets

    Evaluate model quality with real-world metrics (Sharpe, direction accuracy, etc.)

    Support dual-track universes (long-history vs wide-coverage)

    Enable reproducible feature experiments going forward

🛠️ Pipeline Overview

                ┌────────────────────────────┐
                │  sep_base / mid_cap_xx     │
                └────────────┬───────────────┘
                             ▼
     ┌────────────────────────────────────────────┐
     │ filter/filter_universe_by_start_date.py    │
     │ ➤ Outputs: midcap_<year>_universe          │
     └────────────────┬───────────────────────────┘
                      ▼
     ┌────────────────────────────────────────────┐
     │ features/build_feat_matrix_complete.py     │
     │ ➤ Inputs: sep_base + midcap_<year>_universe│
     │ ➤ Outputs: feat_matrix_complete_<year>     │
     └────────────────┬───────────────────────────┘
                      ▼
     ┌────────────────────────────────────────────┐
     │ scripts/generate_targets.py                │
     │ ➤ Computes: ret_1d_f, ret_5d_f, ret_21d_f   │
     │ ➤ Outputs: feat_matrix_targets_<year>      │
     └────────────────┬───────────────────────────┘
                      ▼
     ┌────────────────────────────────────────────┐
     │ models/train_model.py                      │
     │ ➤ Trains Ridge / LGBM on selected targets  │
     │ ➤ Saves: model .pkl, predictions, metrics  │
     └────────────────────────────────────────────┘

🚀 Step-by-Step Usage
1. Filter Modeling Universe by Start Date

python filter/filter_universe_by_start_date.py --year 2008
python filter/filter_universe_by_start_date.py --year 2014

Creates:

    midcap_2008_universe

    midcap_2014_universe
    in DuckDB

2. Build Complete Feature Matrix (3 core features so far)

python features/build_feat_matrix_complete.py --year 2008
python features/build_feat_matrix_complete.py --year 2014

Creates:

    feat_matrix_complete_2008

    feat_matrix_complete_2014
    with:
    log_return, volume_z, price_norm

3. Generate Forward Return Targets

python scripts/generate_targets.py --year 2008
python scripts/generate_targets.py --year 2014

Computes:

    ret_1d_f, ret_5d_f, ret_21d_f

Creates:

    feat_matrix_targets_2008

    feat_matrix_targets_2014

4. Train Ridge or LightGBM Model

Prepare a config file like models/config/ridge_2008.yaml or lgbm_2014.yaml, then:

python models/train_model.py --config models/config/ridge_2008.yaml --year 2008
python models/train_model.py --config models/config/lgbm_2014.yaml --year 2014

Saves to:

    models/output/*.pkl (model)

    predictions_<model>_<year>.csv

    metrics_<model>_<year>.csv

    shap_<model>_<target>_<year>.csv (LGBM only)

📊 Evaluation Metrics

    R² – Fit quality

    MSE – Error magnitude

    Directional Accuracy – How often the model gets up/down direction right

    Sharpe Ratio – Predicted risk-adjusted return (5-day horizon)

🔜 Next Steps (on add-features branch)

    Add additional alpha features (momentum, volatility, fundamentals, etc.)

    Retrain and track SHAP/multi-horizon performance

    Simulate trading strategy performance using predicted ret_5d_f

### [python]

🧪 Full Pipeline Example: 2008 Universe

### 1. Filter tickers active since 2008
python filter/filter_universe_by_start_date.py --year 2008

### 2. Build 3-feature matrix from sep_base + midcap_2008_universe
python features/build_feat_matrix_complete.py --year 2008

### 3. Add forward return targets (ret_1d_f, ret_5d_f, ret_21d_f)
python scripts/generate_targets.py --year 2008

### 4. Train Ridge model on complete + target matrix
python models/train_model.py --config models/config/ridge_2008.yaml --year 2008

🧪 Full Pipeline Example: 2014 Universe

### 1. Filter tickers active since 2014
python filter/filter_universe_by_start_date.py --year 2014

### 2. Build 3-feature matrix from sep_base + midcap_2014_universe
python features/build_feat_matrix_complete.py --year 2014

### 3. Add forward return targets (ret_1d_f, ret_5d_f, ret_21d_f)
python scripts/generate_targets.py --year 2014

### 4. Train LightGBM model on complete + target matrix
python models/train_model.py --config models/config/lgbm_2014.yaml --year 2014

✅ What You’ll Have After Running
Output File	Description
midcap_2008_universe	Tickers with full data since 2008
feat_matrix_complete_2008	Clean feature matrix
feat_matrix_targets_2008	With future return targets
models/output/ridge_ret_*.pkl	Trained Ridge model
predictions_ridge_2008.csv	Model predictions
metrics_ridge_2008.csv	MSE, R², accuracy, Sharpe
And the same for 2014.