import duckdb
import pandas as pd

def export_live_excel(db_path="data/kairos.duckdb", table="live_predictions", output="live_predictions.xlsx"):
    con = duckdb.connect(db_path)
    
    # Fetch the entire table or filter by model/date
    df = con.execute(f"""
        SELECT *
        FROM {table}
        ORDER BY prediction_date DESC, predicted_return DESC
    """).fetchdf()

    # Write to Excel
    df.to_excel(output, index=False)
    print(f"✅ Exported {len(df)} rows to {output}")

def export_batch_excel(db_path="data/kairos.duckdb", table="batch_predictions", output="batch_predictions.xlsx"):
    con = duckdb.connect(db_path)
    
    # Fetch the entire table or filter by model/date
    df = con.execute(f"""
        SELECT *
        FROM {table}
        ORDER BY prediction_date DESC, predicted_return DESC
    """).fetchdf()

    # Write to Excel
    df.to_excel(output, index=False)
    print(f"✅ Exported {len(df)} rows to {output}")

if __name__ == "__main__":
    export_batch_excel()
