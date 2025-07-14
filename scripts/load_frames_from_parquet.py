# scripts/load_frames_from_parquet.py

"""
Rebuild the DuckDB 'frames' table from SSD-backed ticker frame files.

Input:
  - data/ticker_frames/<TICKER>/frames_252/frame_*.parquet
Output:
  - DuckDB table: frames (overwritten)
    Columns: ticker, date, log_return, volume_z, price_norm
"""

import duckdb
import polars as pl
from pathlib import Path
tqdm_enabled = True

# Try tqdm if available
try:
    from tqdm import tqdm
except ImportError:
    tqdm_enabled = False
    tqdm = lambda x, **kwargs: x

DB_PATH = "kairos.duckdb"
FRAMES_ROOT = Path("data/ticker_frames")
WINDOW_FOLDER = "frames_252"


def get_all_frame_paths():
    return sorted(FRAMES_ROOT.glob(f"*/{WINDOW_FOLDER}/frame_*.parquet"))


def load_and_stack_frames(frame_paths):
    all_rows = []
    for fp in tqdm(frame_paths, desc="Loading frames") if tqdm_enabled else frame_paths:
        ticker = fp.parts[-3]
        try:
            df = pl.read_parquet(fp).select(["date", "log_return", "volume_z", "price_norm"])
            df = df.with_columns(pl.lit(ticker).alias("ticker"))
            all_rows.append(df)
        except Exception as e:
            print(f"⚠️ Skipping {fp}: {e}")

    if not all_rows:
        raise RuntimeError("No frame data found!")

    return pl.concat(all_rows, how="vertical")


def main():
    print(f"🔗 Connecting to {DB_PATH}...")
    con = duckdb.connect(DB_PATH)

    frame_paths = get_all_frame_paths()
    print(f"📂 Found {len(frame_paths):,} frame files.")

    print("📥 Loading and stacking frames...")
    df_all = load_and_stack_frames(frame_paths)
    print(f"✅ Loaded {df_all.shape[0]:,} rows across {df_all.select('ticker').n_unique()} tickers")

    print("💾 Overwriting DuckDB 'frames' table...")
    con.execute("DROP TABLE IF EXISTS frames")
    con.register("df_all", df_all.to_pandas())
    con.execute("CREATE TABLE frames AS SELECT ticker, date, log_return, volume_z, price_norm FROM df_all")

    print("🧮 Verifying:")
    res = con.execute("SELECT COUNT(*) AS rows, COUNT(DISTINCT ticker) AS tickers FROM frames").fetchall()
    print(f"✅ {res[0][0]:,} rows across {res[0][1]:,} tickers written to 'frames'")

    con.close()
    print("🎉 Done.")


if __name__ == "__main__":
    main()
