# prepare_training_windows.py (streaming, per-year/ticker)
# ------------------------------------------------------------
# Builds 252-day rolling windows from sep_base in DuckDB
# Streams each (ticker, year) frame to disk to avoid memory blowup

import polars as pl
import duckdb
import argparse
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

# CLI setup
parser = argparse.ArgumentParser()
parser.add_argument("--db", default="data/karios.duckdb", help="Path to DuckDB")
parser.add_argument("--universe", required=True, help="Path to ticker universe CSV")
parser.add_argument("--output-dir", default="data/framed/", help="Where to save frame shards")
parser.add_argument("--window-size", type=int, default=252)
args = parser.parse_args()

# Paths
output_dir = Path(args.output_dir)
frames_dir = output_dir / "frames"
remainder_dir = output_dir / "remainder"
frames_dir.mkdir(parents=True, exist_ok=True)
remainder_dir.mkdir(parents=True, exist_ok=True)

# Load universe
tickers = pl.read_csv(args.universe).select("ticker").to_series().to_list()

# DuckDB connect
conn = duckdb.connect(args.db)

# Per-ticker loop
print("📦 Streaming per-ticker/year frames to disk...")

for ticker in tqdm(tickers):
    query = f"""
        SELECT ticker, date, close, volume
        FROM sep_base
        WHERE ticker = '{ticker}'
        ORDER BY date
    """
    df = conn.execute(query).fetch_df()
    df = pl.from_pandas(df)

    if df.height < args.window_size:
        continue

    # Feature engineering
    df = df.with_columns([
        pl.col("date").cast(pl.Date),
        (pl.col("close") / pl.col("close").shift(1) - 1).alias("log_return"),
        (pl.col("volume") - pl.col("volume").rolling_mean(21)).alias("volume_delta"),
        (pl.col("close") / pl.col("close").mean()).alias("price_norm"),
        pl.col("date").dt.year().alias("year")
    ])

    # Group by year
    for year, df_y in df.group_by("year"):
        df_y = df_y.sort("date")
        year_path = frames_dir / str(year) / ticker
        year_path.mkdir(parents=True, exist_ok=True)

        # Full windows
        for i in range(df_y.height - args.window_size):
            window = df_y.slice(i, args.window_size)
            if window.select(["log_return", "volume_delta", "price_norm"]).null_count().sum_horizontal().item() > 0:
                continue
            X = window.select(["log_return", "volume_delta", "price_norm"]).to_numpy()
            start = window[0, "date"]
            np.save(year_path / f"{ticker}_{start}.npy", X)

        # Final remainder
        remainder = df_y.slice(-args.window_size, args.window_size)
        if remainder.height < args.window_size:
            r_path = remainder_dir / str(year) / ticker
            r_path.mkdir(parents=True, exist_ok=True)
            X_r = remainder.select(["log_return", "volume_delta", "price_norm"]).drop_nulls().to_numpy()
            if X_r.shape[0] > 0:
                np.save(r_path / f"{ticker}_remainder.npy", X_r)

print("✅ All frames streamed to disk.")
