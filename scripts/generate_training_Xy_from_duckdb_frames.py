# scripts/generate_training_Xy_from_duckdb_frames.py

import duckdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import os

def main(db_path="data/karios.duckdb", output_dir="data/model_ready"):
    con = duckdb.connect(db_path)
    print("🔗 Connected to DuckDB")

    print("📊 Fetching all frame rows with next-day closeadj...")
    df = con.execute("""
        WITH next_day AS (
            SELECT
                ticker,
                date,
                closeadj,
                LEAD(closeadj) OVER (PARTITION BY ticker ORDER BY date) AS next_closeadj,
                LEAD(date) OVER (PARTITION BY ticker ORDER BY date) AS next_date
            FROM sep_base
        )
        SELECT
            f.ticker,
            f.frame_path,
            f.date,
            f.log_return,
            f.volume_z,
            f.price_norm,
            n.closeadj AS close_now,
            n.next_closeadj,
            n.next_date
        FROM frames f
        LEFT JOIN next_day n
          ON f.ticker = n.ticker AND f.date = n.date
        WHERE f.frame_path LIKE '%frame_%'
        ORDER BY f.ticker, f.frame_path, f.date
    """).fetchdf()

    print(f"✅ Retrieved {len(df):,} rows")

    X_list, y_list, meta_list = [], [], []
    for (ticker, path), group in tqdm(df.groupby(["ticker", "frame_path"]), desc="Frames"):
        if group.shape[0] != 252:
            continue
        frame = group.sort_values("date")
        features = frame[["log_return", "volume_z", "price_norm"]].to_numpy(dtype=np.float32)
        close_now = frame.iloc[-1]["close_now"]
        close_next = frame.iloc[-1]["next_closeadj"]
        target_date = frame.iloc[-1]["next_date"]

        if pd.isna(close_now) or pd.isna(close_next):
            continue

        y = np.log(close_next / close_now)
        X_list.append(features)
        y_list.append(y)
        meta_list.append({
            "ticker": ticker,
            "start_date": frame.iloc[0]["date"],
            "end_date": frame.iloc[-1]["date"],
            "target_date": target_date
        })

    print(f"✅ Created {len(X_list):,} samples")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    X_all = np.stack(X_list)
    y_all = np.array(y_list, dtype=np.float32)
    meta_df = pd.DataFrame(meta_list)

    np.savez_compressed(os.path.join(output_dir, "X_frames.npz"), X_all)
    np.savez_compressed(os.path.join(output_dir, "y_targets.npz"), y_all)
    meta_df.to_parquet(os.path.join(output_dir, "meta.parquet"))

    print("\n💾 Saved:")
    print(f"📐 X shape: {X_all.shape}")
    print(f"🎯 y shape: {y_all.shape}")
    print(f"🗂️ meta: {meta_df.shape}")

if __name__ == "__main__":
    main()
