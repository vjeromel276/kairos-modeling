# scripts/register_model_features.py

import argparse
import duckdb
import os
import joblib
from pathlib import Path

def create_table_if_needed(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS model_feature_registry (
            model TEXT,
            feature_name TEXT,
            feature_index INTEGER,
            registered_at TIMESTAMP DEFAULT now()
        )
    """)

def register_features(model_name, features, db_path):
    con = duckdb.connect(db_path)
    create_table_if_needed(con)

    # Clear any previous entries for this model
    con.execute("DELETE FROM model_feature_registry WHERE model = ?", (model_name,))

    rows = [(model_name, fname, idx) for idx, fname in enumerate(features)]
    con.executemany("""
        INSERT INTO model_feature_registry (model, feature_name, feature_index)
        VALUES (?, ?, ?)
    """, rows)

    print(f"✅ Registered {len(features)} features for model '{model_name}'")

def get_model_feature_list(model_name, db_path):
    con = duckdb.connect(db_path)
    query = """
        SELECT feature_name
        FROM model_feature_registry
        WHERE model = ?
        ORDER BY feature_index
    """
    result = con.execute(query, (model_name,)).fetchall()
    return [row[0] for row in result]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name (e.g., ridge_2025-07-11)")
    parser.add_argument("--features", nargs="+", required=True, help="List of features used")
    parser.add_argument("--db", default="data/kairos.duckdb", help="Path to DuckDB database")
    args = parser.parse_args()

    register_features(args.model, args.features, args.db)

if __name__ == "__main__":
    main()
