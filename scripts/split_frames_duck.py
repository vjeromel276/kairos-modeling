# split_frames_duck.py
# ------------------------------------------------------------
# Prepare rolling frames per ticker from sep_base table in DuckDB
# Uses same logic as original sep_base.parquet script

import polars as pl
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import duckdb


def compute_features(df):
    df = df.sort("date")
    closeadj = df["closeadj"]
    volume = df["volume"].fill_null(0)

    log_return = pl.Series("log_return", np.log(closeadj[1:] / closeadj[:-1]).to_numpy())
    price_norm = pl.Series("price_norm", (closeadj / closeadj[0]).to_numpy())

    std = volume.std()
    if std is None or std == 0:
        volume_z = pl.Series("volume_z", np.zeros(len(volume)))
    else:
        volume_z = (volume - volume.mean()) / std
        volume_z = volume_z.rename("volume_z")

    df = df.with_columns([
        log_return.extend_constant(None, 1),
        volume_z,
        price_norm
    ])

    return df.select(["ticker", "date", "log_return", "volume_z", "price_norm"])


def split_frames(db_path, ticker, output_dir, window_size):
    conn = duckdb.connect(db_path)
    query = f"""
        SELECT ticker, date, closeadj, volume
        FROM sep_base
        WHERE ticker = '{ticker}'
        ORDER BY date
    """
    df = conn.execute(query).fetch_df()
    conn.close()

    df = pl.from_pandas(df)

    if df.height < window_size:
        return

    df = compute_features(df)
    n = len(df)
    n_frames = n // window_size
    remainder = n % window_size

    ticker_dir = Path(output_dir) / ticker
    frames_dir = ticker_dir / f"frames_{window_size}"
    remainders_dir = ticker_dir / "remainders"
    frames_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n_frames):
        start = i * window_size
        end = start + window_size
        frame = df[start:end]
        frame.write_parquet(frames_dir / f"frame_{i:03d}.parquet")

    if remainder > 0:
        remainders_dir.mkdir(parents=True, exist_ok=True)
        df[n_frames * window_size:].write_parquet(remainders_dir / f"remainder_{window_size}.parquet")


def process_all_tickers(db_path, output_dir, window_size):
    conn = duckdb.connect(db_path)
    tickers = conn.execute("SELECT DISTINCT ticker FROM sep_base").fetchall()
    tickers = [row[0] for row in tickers]
    conn.close()

    print(f"🧠 Processing {len(tickers)} tickers into frames of {window_size} days")

    for ticker in tqdm(tickers, desc="Tickers"):
        split_frames(db_path, ticker, output_dir, window_size)

    print(f"✅ Frames saved under {output_dir}/<TICKER>/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split SEP base (from DuckDB) into per-ticker frames with derived features")
    parser.add_argument("--db", required=True, help="Path to DuckDB database file")
    parser.add_argument("--output-dir", default="data/ticker_frames", help="Output directory for ticker frames")
    parser.add_argument("--window-size", type=int, default=252, help="Length of each frame")
    args = parser.parse_args()

    process_all_tickers(args.db, args.output_dir, args.window_size)
