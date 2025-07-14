# scripts/plot_top_predictions.py

import argparse
import duckdb
import matplotlib.pyplot as plt
import pandas as pd


def plot_top_predictions(db_path, model, run_id=None, top_n=50, table="batch_predictions"):
    con = duckdb.connect(db_path)

    query = f"""
        SELECT ticker, predicted_return
        FROM {table}
        WHERE model = ?
    """
    params = [model]

    if run_id:
        query += " AND run_id = ?"
        params.append(run_id)

    query += " ORDER BY predicted_return DESC LIMIT ?"
    params.append(top_n)

    df = con.execute(query, params).fetchdf()
    plt.figure(figsize=(12, 6))
    plt.bar(df["ticker"], df["predicted_return"], color="green")
    plt.xticks(rotation=90)
    plt.title(f"Top {top_n} Predictions for {model}" + (f" (run_id={run_id})" if run_id else ""))
    plt.ylabel("Predicted Log Return")
    plt.tight_layout()
    output_path = f"outputs/{model}_top_{top_n}" + (f"_{run_id}" if run_id else "") + ".png"
    plt.savefig(output_path)
    print(f"✅ Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/kairos.duckdb", help="Path to DuckDB database")
    parser.add_argument("--model", required=True, help="Model name to filter predictions")
    parser.add_argument("--run-id", default=None, help="Optional run_id to isolate predictions")
    parser.add_argument("--top-n", type=int, default=50, help="Number of top predictions to plot")
    parser.add_argument("--table", default="batch_predictions", help="DuckDB table to query from")
    args = parser.parse_args()

    plot_top_predictions(args.db, args.model, args.run_id, args.top_n, args.table)


if __name__ == "__main__":
    main()
