#!/usr/bin/env python3
"""
scripts/predict_live.py

Generates live predictions for most recent date per ticker
using trained model and filtered live universe.
Also saves the live universe snapshot to a table for traceability.
"""

import duckdb
import pandas as pd
import argparse
import joblib
from pathlib import Path
from datetime import datetime
import yaml

# ------------------------
# Argument Parsing
# ------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--model-file', required=True, help="Path to trained .pkl model")
parser.add_argument('--config', required=True, help="Path to YAML config")
parser.add_argument('--year', type=int, required=True, help="Universe year (e.g. 2008)")
parser.add_argument('--tag', type=str, required=True, help="Tag name for this live prediction run")
args = parser.parse_args()

# ------------------------
# Load Config + Model
# ------------------------
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

features = config['features']
targets = config['targets']
model_type = config['model']
model = joblib.load(args.model_file)

# ------------------------
# Connect to DuckDB
# ------------------------
DB_PATH = 'data/kairos.duckdb'
TABLE_NAME = f'feat_matrix_complete_{args.year}'
LIVE_UNIVERSE_TABLE = 'midcap_live_universe'
con = duckdb.connect(DB_PATH)

# ------------------------
# Build Filtered Universe Snapshot
# ------------------------
print("🧠 Building tagged live universe snapshot...")
universe_df = con.execute(f"""
     WITH latest_date AS (
        SELECT MAX(date) AS max_date FROM sep_base
        ),
        base_universe AS (
        SELECT ticker
        FROM ticker_metadata_view
        WHERE
            scalemarketcap IN ('4 - Mid', '5 - Large', '6 - Mega') AND
            LOWER(category) LIKE '%common stock%' AND
            volumeavg1m >= 2000000
        ),
        long_history AS (
        SELECT ticker
        FROM sep_base
        GROUP BY ticker
        HAVING COUNT(DISTINCT date) >= 252
        ),
        has_latest_date AS (
        SELECT DISTINCT ticker
        FROM sep_base
        WHERE date = (SELECT max_date FROM latest_date)
        )
        SELECT b.ticker, (SELECT max_date FROM latest_date) AS as_of_date, '{args.tag}' AS tag
        FROM base_universe b
        JOIN long_history l USING (ticker)
        JOIN has_latest_date h USING (ticker)
""").fetchdf()

con.execute(f"DROP TABLE IF EXISTS {LIVE_UNIVERSE_TABLE}")
con.register("universe_df", universe_df)
con.execute(f"CREATE TABLE {LIVE_UNIVERSE_TABLE} AS SELECT * FROM universe_df")

# ------------------------
# Pull latest features for that universe with SAFF fallback
# ------------------------
print(f"📥 Pulling latest features from {TABLE_NAME} with SAFF...")
df = con.execute(f"""
    WITH latest_feat AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY date DESC) AS rn
        FROM {TABLE_NAME}
    )
    SELECT f.*
    FROM latest_feat f
    JOIN {LIVE_UNIVERSE_TABLE} u USING (ticker)
    WHERE rn = 1
""").fetchdf()

# Optional: Apply SAFF-like fallback manually
for col in features:
    if col not in df.columns:
        df[col] = None

df = df.dropna(subset=features, how='any')

# ------------------------
# Run Predictions
# ------------------------
X = df[features]
print(f"🔮 Predicting {len(X)} live signals using {model_type}...")
y_pred = model.predict(X)

# ------------------------
# Format Output
# ------------------------
pred_df = df[["ticker", "date"]].copy()
for i, t in enumerate(targets):
    pred_df[f"{t}_pred"] = y_pred[:, i]

pred_df["model"] = Path(args.config).stem
pred_df["run_date"] = datetime.utcnow()

# ------------------------
# Save Predictions
# ------------------------
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
# ------------------------