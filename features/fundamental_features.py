#!/usr/bin/env python3
"""
DEBUG version: Extract valuation features from SHARADAR DAILY table.
Uses SAFF (Smart Attenuated Forward Fill) logic from feat_utils to fill missing fundamentals.
"""

import duckdb
import pandas as pd
import numpy as np
import argparse

# import our utility functions
from features.features_utils import expand_and_saff


def compute_fundamental_features(con):
    # 1) Load raw SHARADAR fundamentals
    print("📥 Loading raw SHARADAR daily fundamental data...")
    df_raw = con.execute(
        """
        SELECT ticker, date, marketcap, ev, evebit, evebitda, pe, pb, ps
        FROM daily
        WHERE ticker IS NOT NULL
        ORDER BY ticker, date
        """
    ).fetchdf()
    print(f"✅ Initial load complete: {len(df_raw):,} rows of raw fundamentals.")

    # 2) Define which columns to SAFF-fill
    value_cols = ["marketcap", "ev", "evebit", "evebitda", "pe", "pb", "ps"]
    ticker_col = "ticker"
    date_col = "date"

    # 3) Expand to full ticker-date grid and apply SAFF
    print("🔍 Expanding to full ticker-date range and applying SAFF...")
    df = expand_and_saff(
        con=con,
        sparse_df=df_raw, # type: ignore
        base_table="sep_base_common",
        value_cols=value_cols,
        ticker_col=ticker_col,
        date_col=date_col
    )
    print(f"✅ Expanded & SAFF applied: {df.shape[0]:,} rows.")

    # 4) Filter out extreme or invalid values
    # df = df[
    #     (df.marketcap > 0) &
    #     (df.pe > 0) & (df.pe < 100) &
    #     (df.pb > 0) & (df.pb < 20) &
    #     (df.ps > 0) & (df.ps < 20) &
    #     (df.evebitda > 0) & (df.evebitda < 1e6) &
    #     (df.evebit > 0) & (df.evebit < 1e6) &
    #     (df.ev > 0)
    # ]
    # print(f"✅ Filtering complete: {len(df):,} rows remain after filtering.")

    # 4) Winsorize our valuation multiples so we DON’T drop entire rows (e.g. TSLA)
    from features.features_utils import winsorize

    # only these columns need capping
    mult_cols = ["pe","pb","ps","evebitda","evebit"]
    # do a per-date winsorization so the universe’s changing distribution is respected
    df = winsorize(df, cols=mult_cols, lower_q=0.01, upper_q=0.99, by_date=True, date_col="date")
    print("✅ Winsorized valuation multiples at [1%,99%] per date.")

    # 5) Feature engineering
    df['log_marketcap'] = np.log(df['marketcap'])
    df['ev_to_ebitda']  = df['ev'] / df['evebitda']
    df['ev_to_ebit']    = df['ev'] / df['evebit']
    df['value_composite'] = (
        1/df['pe'].replace(0, np.nan) +
        1/df['pb'].replace(0, np.nan) +
        1/df['ps'].replace(0, np.nan)
    )

    # 6) Compute per-date z-scores
    z_cols = [
        'log_marketcap', 'pe', 'pb', 'ps',
        'ev_to_ebitda', 'ev_to_ebit', 'value_composite'
    ]
    print(f"🧪 Computing per-date z-scores over {df['date'].nunique():,} dates...")

    dfs = []
    for date, group in df.groupby('date'):
        g = group.copy()
        for col in z_cols:
            mu, sigma = g[col].mean(), g[col].std()
            g[f"{col}_z"] = (g[col] - mu) / sigma if sigma else np.nan
        dfs.append(g)
    df_feat = pd.concat(dfs, ignore_index=True).dropna()

    # 7) Select final columns
    final_cols = ['ticker', 'date'] + z_cols + [f"{c}_z" for c in z_cols]
    return df_feat[final_cols]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_fundamentals")

    df_feat = compute_fundamental_features(con)
    con.execute("CREATE TABLE feat_fundamentals AS SELECT * FROM df_feat")
    print(f"✅ Saved {len(df_feat):,} rows to feat_fundamentals.")

if __name__ == "__main__":
    main()
