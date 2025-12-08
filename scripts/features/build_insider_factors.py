#!/usr/bin/env python3
"""
build_insider_factors.py
========================
Build insider trading signal from the Sharadar 'sf2' table.

Based on Phase 4 Investigation findings:
- Insider net buying: IC +0.0049
- 11.2M rows of insider transaction data available

Transaction codes in SF2:
- P = Open market purchase
- S = Open market sale
- A = Grant/award
- D = Disposition to issuer
- G = Gift
- F = Payment of exercise price
- M = Exercise of derivative
- X = Exercise of out-of-money derivative
- C = Conversion of derivative
- W = Acquisition or disposition by will/inheritance

Focus on P (purchases) and S (sales) for cleanest signal.

Output table: feat_insider
Columns: ticker, date, buy_value_30d, sell_value_30d, net_buy_value_30d,
         buy_count_30d, sell_count_30d, net_buy_signal, insider_composite_z
"""

import argparse
import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def zscore_winsorize(s: pd.Series, clip: float = 3.0) -> pd.Series:
    """Z-score with winsorization at +/- clip."""
    mean = s.mean()
    std = s.std()
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    z = (s - mean) / std
    return z.clip(-clip, clip)

def main():
    parser = argparse.ArgumentParser(description="Build insider factors from SF2")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--lookback-days", type=int, default=30, help="Lookback window for aggregation")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("BUILD INSIDER FACTORS (From Sharadar SF2)")
    print(f"{'='*60}\n")

    con = duckdb.connect(args.db)

    # Check SF2 table
    print("Checking sf2 table...")
    
    # Get schema
    schema = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'sf2'
    """).fetchall()
    cols = [c[0] for c in schema]
    print(f"  Columns: {cols}")

    count = con.execute("SELECT COUNT(*) FROM sf2").fetchone()[0]
    print(f"  Total rows in sf2: {count:,}")

    # Check transaction codes
    print("\nTransaction code distribution:")
    tx_dist = con.execute("""
        SELECT transactioncode, COUNT(*) as cnt
        FROM sf2
        WHERE transactioncode IS NOT NULL
        GROUP BY transactioncode
        ORDER BY cnt DESC
    """).fetchdf()
    print(tx_dist.to_string())

    # Get date range
    date_range = con.execute("""
        SELECT MIN(filingdate) as min_date, MAX(filingdate) as max_date
        FROM sf2
    """).fetchdf()
    print(f"\nDate range: {date_range['min_date'].iloc[0]} to {date_range['max_date'].iloc[0]}")

    # Aggregate insider transactions by ticker and filing date
    # Focus on P (purchases) and S (sales)
    print(f"\nAggregating insider transactions (lookback={args.lookback_days} days)...")
    
    # First, get all unique trading dates from SEP (to expand to)
    print("  Getting trading dates from sep...")
    trading_dates = con.execute("""
        SELECT DISTINCT date 
        FROM sep 
        WHERE date >= '2000-01-01'
        ORDER BY date
    """).fetchdf()
    print(f"  Found {len(trading_dates):,} trading dates")

    # Aggregate SF2 by ticker and filing date
    print("  Aggregating SF2 transactions...")
    sf2_agg = con.execute("""
        SELECT 
            ticker,
            filingdate as date,
            SUM(CASE WHEN transactioncode = 'P' THEN ABS(transactionvalue) ELSE 0 END) AS buy_value,
            SUM(CASE WHEN transactioncode = 'S' THEN ABS(transactionvalue) ELSE 0 END) AS sell_value,
            SUM(CASE WHEN transactioncode = 'P' THEN 1 ELSE 0 END) AS buy_count,
            SUM(CASE WHEN transactioncode = 'S' THEN 1 ELSE 0 END) AS sell_count
        FROM sf2
        WHERE transactioncode IN ('P', 'S')
          AND transactionvalue IS NOT NULL
          AND filingdate >= '2000-01-01'
        GROUP BY ticker, filingdate
    """).fetchdf()
    print(f"  Aggregated to {len(sf2_agg):,} ticker-date combinations")

    # Get all tickers that have insider data
    tickers = sf2_agg['ticker'].unique()
    print(f"  Unique tickers with insider data: {len(tickers):,}")

    # For each ticker, expand to daily and compute rolling sums
    print("  Computing rolling sums (this may take a while)...")
    
    # Create a date grid for all tickers
    # This is memory-intensive, so we'll do it in chunks
    
    results = []
    chunk_size = 500
    
    for i in range(0, len(tickers), chunk_size):
        chunk_tickers = tickers[i:i+chunk_size]
        
        # Get insider data for this chunk
        chunk_insider = sf2_agg[sf2_agg['ticker'].isin(chunk_tickers)].copy()
        
        # For each ticker, expand to daily calendar and compute rolling sums
        for ticker in chunk_tickers:
            ticker_data = chunk_insider[chunk_insider['ticker'] == ticker].copy()
            
            if len(ticker_data) == 0:
                continue
            
            # Get date range for this ticker
            min_date = ticker_data['date'].min()
            max_date = ticker_data['date'].max()
            
            # Filter trading dates to this range
            ticker_dates = trading_dates[
                (trading_dates['date'] >= min_date) & 
                (trading_dates['date'] <= max_date)
            ].copy()
            ticker_dates['ticker'] = ticker
            
            # Merge with insider data
            merged = ticker_dates.merge(ticker_data, on=['ticker', 'date'], how='left')
            merged = merged.fillna(0)
            
            # Compute rolling sums
            merged = merged.sort_values('date')
            merged['buy_value_30d'] = merged['buy_value'].rolling(args.lookback_days, min_periods=1).sum()
            merged['sell_value_30d'] = merged['sell_value'].rolling(args.lookback_days, min_periods=1).sum()
            merged['buy_count_30d'] = merged['buy_count'].rolling(args.lookback_days, min_periods=1).sum()
            merged['sell_count_30d'] = merged['sell_count'].rolling(args.lookback_days, min_periods=1).sum()
            
            # Net values
            merged['net_buy_value_30d'] = merged['buy_value_30d'] - merged['sell_value_30d']
            
            # Net buy signal: sign of net value
            merged['net_buy_signal'] = np.sign(merged['net_buy_value_30d'])
            
            results.append(merged[['ticker', 'date', 'buy_value_30d', 'sell_value_30d', 
                                   'net_buy_value_30d', 'buy_count_30d', 'sell_count_30d', 
                                   'net_buy_signal']])
        
        print(f"    Processed {min(i+chunk_size, len(tickers)):,} / {len(tickers):,} tickers")

    # Combine all results
    print("\nCombining results...")
    df = pd.concat(results, ignore_index=True)
    print(f"  Total rows: {len(df):,}")

    # Compute cross-sectional z-scores by date
    print("Computing cross-sectional z-scores...")
    
    def zscore_group(group):
        # Z-score the net buy value (log-transformed for better distribution)
        net_val = group['net_buy_value_30d']
        # Log transform (handle negatives)
        log_net = np.sign(net_val) * np.log1p(np.abs(net_val))
        group['insider_composite_z'] = zscore_winsorize(log_net)
        return group
    
    df = df.groupby('date', group_keys=False).apply(zscore_group)

    # Fill NaN z-scores with 0 (no insider activity = neutral)
    df['insider_composite_z'] = df['insider_composite_z'].fillna(0)

    # Write to database
    print("\nWriting to database...")
    con.execute("DROP TABLE IF EXISTS feat_insider")
    con.execute("""
        CREATE TABLE feat_insider (
            ticker VARCHAR,
            date DATE,
            buy_value_30d DOUBLE,
            sell_value_30d DOUBLE,
            net_buy_value_30d DOUBLE,
            buy_count_30d DOUBLE,
            sell_count_30d DOUBLE,
            net_buy_signal DOUBLE,
            insider_composite_z DOUBLE
        )
    """)
    
    # Insert in batches
    batch_size = 500000
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        con.execute("INSERT INTO feat_insider SELECT * FROM batch")
        print(f"  Inserted {min(i+batch_size, len(df)):,} / {len(df):,}")

    # Create indexes
    print("Creating indexes...")
    con.execute("CREATE INDEX IF NOT EXISTS idx_insider_ticker_date ON feat_insider(ticker, date)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_insider_date ON feat_insider(date)")

    # Verify
    final_count = con.execute("SELECT COUNT(*) FROM feat_insider").fetchone()[0]
    print(f"\n✓ Created feat_insider with {final_count:,} rows")

    # Summary stats
    print("\nSummary statistics:")
    stats = con.execute("""
        SELECT 
            MIN(date) as min_date,
            MAX(date) as max_date,
            COUNT(DISTINCT ticker) as n_tickers,
            AVG(insider_composite_z) as avg_composite,
            STDDEV(insider_composite_z) as std_composite,
            AVG(net_buy_signal) as avg_signal
        FROM feat_insider
    """).fetchdf()
    print(stats.to_string())

    con.close()
    print(f"\n{'='*60}")
    print("DONE - Insider factors built successfully")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
