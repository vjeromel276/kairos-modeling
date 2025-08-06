#!/usr/bin/env python3
import os
import argparse
import duckdb
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Build multi-horizon windows via Parquet shards + final DuckDB merge"
    )
    p.add_argument("--db-path",       default="data/kairos.duckdb")
    p.add_argument("--shards-dir",    default="shards")
    p.add_argument("--window-size",   type=int, default=252)
    p.add_argument("--max-horizon",   type=int, default=21)
    p.add_argument("--workers",       type=int, default=os.cpu_count())
    p.add_argument("--tickers",       type=str,
                   help="Comma-separated list. Default=all in sep_base_common.")
    return p.parse_args()

def get_tickers(conn, tickers_arg):
    if tickers_arg:
        return [t.strip() for t in tickers_arg.split(",")]
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM sep_base_common"
    ).fetchall()]

def process_ticker(ticker, db_path, shards_dir, window_size, max_horizon):
    # 1) Load history in read-only mode
    conn = duckdb.connect(database=db_path, read_only=True)
    df = conn.execute(f"""
        SELECT
          CAST(date AS DATE) AS date,
          open, high, low, close, volume, closeadj
        FROM sep_base_common
        WHERE ticker = '{ticker}'
        ORDER BY date
    """).df()
    conn.close()

    n_rows    = len(df)
    n_windows = n_rows - window_size - max_horizon + 1
    if n_windows < 1:
        return  # not enough history

    # 2) Build lists of columns
    starts = []
    ends   = []
    r1s    = []
    r5s    = []
    r21s   = []
    feats  = []

    for i in range(n_windows):
        win = df.iloc[i : i + window_size]
        start = win.date.iloc[0]
        end_i = i + window_size - 1
        end   = df.date.iloc[end_i]

        # forward returns
        p_t   = df.closeadj.iloc[end_i]
        p1    = df.closeadj.iloc[end_i + 1]
        p5    = df.closeadj.iloc[end_i + 5]
        p21   = df.closeadj.iloc[end_i + max_horizon]
        r1s.append( np.log(p1 / p_t) )
        r5s.append( np.log(p5 / p_t) )
        r21s.append(np.log(p21/ p_t) )

        # record dates
        starts.append(start)
        ends.append(end)

        # features → flatten
        feats.append(
            win[["open","high","low","close","volume","closeadj"]]
              .to_numpy().flatten().tolist()
        )

    # 3) Write shard (no DB writes here)
    df_shard = pd.DataFrame({
        "ticker":      [ticker]*n_windows,
        "start_date":  starts,
        "end_date":    ends,
        "ret_1d_f":    r1s,
        "ret_5d_f":    r5s,
        "ret_21d_f":   r21s,
        "features":    feats
    })
    os.makedirs(shards_dir, exist_ok=True)
    out_path = os.path.join(shards_dir, f"mh_windows_{ticker}.parquet")
    df_shard.to_parquet(out_path)

def main():
    args = parse_args()
    db_path     = args.db_path
    shards_dir  = args.shards_dir
    window_size = args.window_size
    max_horizon = args.max_horizon
    workers     = args.workers or os.cpu_count()

    # 1) discover tickers in read-only mode
    conn    = duckdb.connect(database=db_path, read_only=True)
    tickers = get_tickers(conn, args.tickers)
    conn.close()

    # 2) parallel shard export
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_ticker,
                ticker,
                db_path,
                shards_dir,
                window_size,
                max_horizon
            ): ticker
            for ticker in tickers
        }

        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc="Sharding tickers"):
            ticker = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[ERROR] {ticker}: {e}")

    # 3) single‐writer merge into DuckDB (now safe)
    conn = duckdb.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS mh_windows;")
    conn.execute(f"""
        CREATE TABLE mh_windows AS
        SELECT * FROM read_parquet('{shards_dir}/*.parquet');
    """)
    conn.close()

    # 4) cleanup shards
    for fn in os.listdir(shards_dir):
        if fn.endswith(".parquet"):
            os.remove(os.path.join(shards_dir, fn))

    print("✅ Done: mh_windows is rebuilt and shards cleaned up.")

if __name__ == "__main__":
    main()
