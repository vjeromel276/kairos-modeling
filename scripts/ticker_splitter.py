import duckdb
import pandas as pd
import numpy as np
import tqdm

# 1) Connect to your DuckDB and read AAPL’s history once
conn = duckdb.connect("data/kairos.duckdb")
df = conn.execute("""
    SELECT 
      CAST(date AS DATE) AS date,
      open, high, low, close, volume, closeadj
    FROM sep_base_common
    WHERE ticker = 'AAPL'
    ORDER BY date
""").df()

window_size = 252
max_horizon = 21  # for ret_21d

# 2) Prepare the target table for windows (with targets)
conn.execute("""
    CREATE TABLE IF NOT EXISTS mh_windows (
      ticker      VARCHAR,
      start_date  DATE,
      end_date    DATE,
      ret_1d_f    DOUBLE,
      ret_5d_f    DOUBLE,
      ret_21d_f   DOUBLE,
      features    DOUBLE[]
    )
""")

# 3) Slide and stream each window into DuckDB, computing targets
n_rows = len(df)
n_windows = n_rows - window_size - max_horizon + 1

for i in tqdm.tqdm(range(n_windows), desc="Processing windows"):
    win = df.iloc[i : i + window_size]
    start = win["date"].iloc[0]
    end_idx = i + window_size - 1
    end   = df["date"].iloc[end_idx]

    # compute forward prices
    price_t   = df["closeadj"].iloc[end_idx]
    price_t1  = df["closeadj"].iloc[end_idx + 1]
    price_t5  = df["closeadj"].iloc[end_idx + 5]
    price_t21 = df["closeadj"].iloc[end_idx + max_horizon]

    # compute forward log‐returns
    ret_1d_f  = np.log(price_t1  / price_t)
    ret_5d_f  = np.log(price_t5  / price_t)
    ret_21d_f = np.log(price_t21 / price_t)

    # flatten features
    feat_array = win[["open","high","low","close","volume","closeadj"]].to_numpy().flatten().tolist()

    conn.execute(
        """
        INSERT INTO mh_windows 
          (ticker, start_date, end_date, ret_1d_f, ret_5d_f, ret_21d_f, features)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ["AAPL", start, end, float(ret_1d_f), float(ret_5d_f), float(ret_21d_f), feat_array]
    )

conn.close()
