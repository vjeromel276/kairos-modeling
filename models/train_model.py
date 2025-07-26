#!/usr/bin/env python3
"""
Enhanced model training harness for Kairos pipeline.
Trains on feat_matrix_targets_<year>, evaluates performance,
outputs predictions (with actuals), and computes SHAP values for LightGBM.
"""

import duckdb
import argparse
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import yaml
import joblib
import os
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------
# Argument Parsing
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to YAML config')
parser.add_argument('--year', type=int, required=True, help='Universe year (e.g. 2008, 2014)')
args = parser.parse_args()

# ------------------------
# Load Config
# ------------------------
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

model_type = config['model']
targets = config['targets']
features = config['features']
params = config.get('params', {})

# ------------------------
# Load Data
# ------------------------
DB_PATH = 'data/kairos.duckdb'
TABLE_NAME = f'feat_matrix_targets_{args.year}'
print(f"📦 Loading {TABLE_NAME} from DuckDB...")
con = duckdb.connect(DB_PATH)
df = con.execute(f"SELECT * FROM {TABLE_NAME}").fetchdf()

# ------------------------
# Prepare Training Data
# ------------------------
df = df.dropna(subset=features + targets)
df = df.sort_values("date")

# Time-based validation split
split_date = df["date"].quantile(0.8)
df_train = df[df["date"] <= split_date]
df_val   = df[df["date"] > split_date]

X_train, y_train = df_train[features], df_train[targets]
X_val, y_val     = df_val[features], df_val[targets]

print(f"🧪 Training on {len(X_train)} rows, validating on {len(X_val)} rows")

# ------------------------
# Model Selection
# ------------------------
if model_type == 'ridge':
    params.pop('normalize', None)
    base_model = Ridge(**params)
elif model_type == 'lgbm':
    base_model = LGBMRegressor(**params)
elif model_type == 'mlp':
    base_model = MLPRegressor(**params)
elif model_type == 'xgb':
    base_model = XGBRegressor(**params)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

model = MultiOutputRegressor(base_model) # type: ignore
# Final safety cleaning: drop any inf or NaNs
X_train = X_train.replace([np.inf, -np.inf], np.nan).dropna()
y_train = y_train.loc[X_train.index]

X_val = X_val.replace([np.inf, -np.inf], np.nan).dropna()
y_val = y_val.loc[X_val.index]
# ------------------------
# Train Model
# ------------------------
print(f"🔍 Training {model_type} model...")
model.fit(X_train, y_train)

# ------------------------
# Evaluate
# ------------------------
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
r2 = r2_score(y_val, y_pred, multioutput='raw_values')
acc = [(np.sign(y_val[t]) == np.sign(y_pred[:, i])).mean() for i, t in enumerate(targets)] # type: ignore
sharpe = (y_pred.mean(axis=0) / y_pred.std(axis=0)) * np.sqrt(252 / 5) # type: ignore

eval_df = pd.DataFrame({
    'target': targets,
    'mse': mse,
    'r2': r2,
    'directional_accuracy': acc,
    'sharpe': sharpe
})
print("📊 Evaluation:")
print(eval_df)

# ------------------------
# Save Outputs
# ------------------------
OUTPUT_DIR = 'models/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_name = f"{model_type}_{'_'.join(targets)}_{args.year}.pkl"
pred_name = f"predictions_{model_type}_{args.year}.csv"
metrics_name = f"metrics_{model_type}_{args.year}.csv"

# Save predictions with actuals
pred_df = df_val.loc[X_val.index, ["ticker", "date"] + targets].copy()
for i, t in enumerate(targets):
    pred_df[f"{t}_pred"] = y_pred[:, i] # type: ignore

joblib.dump(model, os.path.join(OUTPUT_DIR, model_name))
pred_df.to_csv(os.path.join(OUTPUT_DIR, pred_name), index=False)
eval_df.to_csv(os.path.join(OUTPUT_DIR, metrics_name), index=False)

print(f"✅ Model saved to {model_name}")
print(f"✅ Predictions saved to {pred_name}")
print(f"✅ Metrics saved to {metrics_name}")

# ------------------------
# SHAP for LightGBM
# ------------------------
if model_type == 'lgbm':
    import shap
    print("🔍 Computing SHAP values...")

    shap_values = []
    for i, est in enumerate(model.estimators_):
        explainer = shap.TreeExplainer(est)
        sv = explainer.shap_values(X_val)
        shap_values.append(sv)

        shap_df = pd.DataFrame({
            'feature': X_val.columns,
            'mean_abs_shap': np.abs(sv).mean(axis=0)
        }).sort_values(by='mean_abs_shap', ascending=False).head(20)

        shap_df.to_csv(f"{OUTPUT_DIR}/shap_{model_type}_{targets[i]}_{args.year}.csv", index=False)
        print(f"✅ SHAP saved: shap_{model_type}_{targets[i]}_{args.year}.csv")
