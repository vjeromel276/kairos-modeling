#!/usr/bin/env python3
"""
Generates live predictions for most recent date available per ticker.
Uses trained model (.pkl) and latest features from DuckDB.
Saves to live_predictions table.
"""

import duckdb
import pandas as pd
import argparse
import joblib
from pathlib import Path
from datetime import datetime
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--model-file', required=True, help="Path to trained .pkl model")
parser.add_argument('--config', required=True, help="Path to YAML config")
parser.add_argument('--year', type=int, required=True, help="Universe year (e.g. 2008)")
args = parser.parse_args()

# Load config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

features = config['features']
targets = config['targets']
model_type = config['model']

# Load model
model = joblib.load(args.model_file)

# Connect to DuckDB
DB_PATH = 'data/kairos.duckdb'
TABLE_NAME = f'feat_matrix_complete_{args.year}'
con = duckdb.connect(DB_PATH)

# Get most recent row per ticker
print(f"📥 Pulling latest features from {TABLE_NAME}...")
query = f"""
    SELECT *
    FROM (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
        FROM {TABLE_NAME}
    )
    WHERE rn = 1
"""
df = con.execute(query).fetchdf()

# Clean and subset
df = df.dropna(subset=features)
X = df[features]

# Predict
print(f"🔮 Predicting {len(X)} live signals using {model_type}...")
y_pred = model.predict(X)

# Prepare output
pred_df = df[["ticker", "date"]].copy()
for i, t in enumerate(targets):
    pred_df[f"{t}_pred"] = y_pred[:, i]

pred_df["model"] = Path(args.config).name.replace(".yaml", "")
pred_df["run_date"] = datetime.utcnow()

# Save to DuckDB
print("💾 Saving to DuckDB table: live_predictions...")
con.execute("""
    CREATE TABLE IF NOT EXISTS live_predictions (
        ticker TEXT,
        date TIMESTAMP,
        ret_1d_f_pred DOUBLE,
        ret_5d_f_pred DOUBLE,
        ret_21d_f_pred DOUBLE,
        model TEXT,
        run_date TIMESTAMP
    )
""")

con.register("pred_df", pred_df)
con.execute("""
    INSERT INTO live_predictions (
        ticker, date,
        ret_1d_f_pred, ret_5d_f_pred, ret_21d_f_pred,
        model, run_date
    )
    SELECT ticker, date,
           ret_1d_f_pred, ret_5d_f_pred, ret_21d_f_pred,
           model, run_date
    FROM pred_df
""")

print("✅ Live predictions saved.")
