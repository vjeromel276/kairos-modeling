#!/usr/bin/env python3
"""
Enhanced model training harness for Kairos.
Includes: train/val split, R², MSE, directional accuracy, Sharpe ratio.
"""

import duckdb
import argparse
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
# Output Directory
# ------------------------
OUTPUT_DIR = 'models/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------
# Prepare Training Data
# ------------------------
df = df.dropna(subset=features + targets)
df = df.sort_values("date")

# Time-based split: last 20% for validation
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
    base_model = Ridge(**params)
elif model_type == 'lgbm':
    base_model = LGBMRegressor(**params)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

model = MultiOutputRegressor(base_model) # type: ignore
model.fit(X_train, y_train)

# ------------------------
# Evaluation
# ------------------------
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred, multioutput='raw_values')
r2 = r2_score(y_val, y_pred, multioutput='raw_values')

# Directional Accuracy
acc = [
    (np.sign(y_val[t]) == np.sign(y_pred[:, i])).mean() # type: ignore
    for i, t in enumerate(targets)
]

# Sharpe Ratio
rets = y_val.values
pred_rets = y_pred
excess = pred_rets - 0.0  # type: ignore # Assume risk-free = 0
sharpe = (excess.mean(axis=0) / excess.std(axis=0)) * np.sqrt(252 / 5)  # Annualized, assuming 5-day hold

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
# SHAP Value Analysis (LightGBM only)
# ------------------------
if model_type == 'lgbm':
    import shap
    print("🔍 Computing SHAP values...")

    explainer = shap.Explainer(model.estimators_[0], X_val)
    shap_values = [explainer(X_val) for model in model.estimators_]

    # Save top-20 mean absolute SHAP values per target
    for i, (target, sv) in enumerate(zip(targets, shap_values)):
        shap_df = pd.DataFrame({
            'feature': X_val.columns,
            'mean_abs_shap': np.abs(sv.values).mean(axis=0)
        }).sort_values(by='mean_abs_shap', ascending=False).head(20)

        shap_df.to_csv(os.path.join(OUTPUT_DIR, f"shap_{model_type}_{target}_{args.year}.csv"), index=False)
        print(f"✅ SHAP values saved for {target}")

# ------------------------
# Save Outputs
# ------------------------

model_name = f"{model_type}_{'_'.join(targets)}_{args.year}.pkl"
pred_name = f"predictions_{model_type}_{args.year}.csv"
metrics_name = f"metrics_{model_type}_{args.year}.csv"

joblib.dump(model, os.path.join(OUTPUT_DIR, model_name))
pd.DataFrame(y_pred, columns=[f'{t}_pred' for t in targets]).to_csv(os.path.join(OUTPUT_DIR, pred_name), index=False) # type: ignore
eval_df.to_csv(os.path.join(OUTPUT_DIR, metrics_name), index=False)

print(f"✅ Model saved to {model_name}")
print(f"✅ Predictions saved to {pred_name}")
print(f"✅ Metrics saved to {metrics_name}")
