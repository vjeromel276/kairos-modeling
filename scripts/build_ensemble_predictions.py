#!/usr/bin/env python3
"""
Aggregates live_predictions into ensemble predictions for a given date.
Uses average method to combine ret_5d_f_pred across models.
Saves to DuckDB table: ensemble_predictions.
"""

import duckdb
import argparse
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--date", required=True, help="Prediction date (YYYY-MM-DD)")
parser.add_argument("--method", default="avg", help="Ensembling method (default: avg)")
args = parser.parse_args()

target_date = args.date
method = args.method
run_date = datetime.utcnow()

con = duckdb.connect("data/kairos.duckdb")

# Step 1: Pull all live predictions for given date
query = f"""
    SELECT ticker, date, model, ret_5d_f_pred
    FROM live_predictions
    WHERE date = DATE '{target_date}'
"""
df = con.execute(query).fetchdf()

if df.empty:
    raise ValueError(f"No predictions found for {target_date} in live_predictions")

# Step 2: Aggregate by ticker
agg_df = df.groupby(["ticker", "date"]).agg({"ret_5d_f_pred": "mean"}).reset_index()
agg_df["rank"] = agg_df["ret_5d_f_pred"].rank(method="dense", ascending=False).astype(int)
agg_df["run_date"] = run_date
agg_df["method"] = method
agg_df["models_used"] = ",".join(sorted(df["model"].unique()))

# Step 3: Create output table if needed
con.execute("""
    CREATE TABLE IF NOT EXISTS ensemble_predictions (
        ticker TEXT,
        date TIMESTAMP,
        ensemble_5d_pred DOUBLE,
        rank INT,
        run_date TIMESTAMP,
        method TEXT,
        models_used TEXT
    )
""")

# Step 4: Insert into table
con.register("agg_df", agg_df.rename(columns={"ret_5d_f_pred": "ensemble_5d_pred"}))
con.execute("""
    INSERT INTO ensemble_predictions (
        ticker, date, ensemble_5d_pred, rank, run_date, method, models_used
    )
    SELECT ticker, date, ensemble_5d_pred, rank, run_date, method, models_used
    FROM agg_df
""")

print(f"✅ Ensemble predictions for {target_date} saved to DuckDB.")
