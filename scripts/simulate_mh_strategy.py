#!/usr/bin/env python3
"""
simulate_mh_strategy.py

Backtest a top-K multi-horizon strategy:
- On each prediction date (mh_meta_252.start_date), rank tickers by model-predicted ret_5d.
- Go long top K tickers, equally weighted.
- Hold H days (default 5), then record realized ret_5d.
- Compute daily P&L series and performance stats.
"""

import argparse
import duckdb
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta

def load_data(db_path, window):
    con = duckdb.connect(db_path)
    # metadata with start_date, end_date, ticker
    meta = con.execute(f"SELECT * FROM mh_meta_{window}").df()
    # actual forward returns
    y = con.execute(f"SELECT * FROM mh_y_{window}").df()
    X = con.execute(f"SELECT * FROM mh_X_{window}").df()
    con.close()
    # merge into one DataFrame
    df = meta.copy()
    df[['ret_1d_f','ret_5d_f','ret_21d_f']] = y[['ret_1d_f','ret_5d_f','ret_21d_f']].values
    return df, X

def predict(df, X, model):
    preds = model.predict(X.values)
    df = df.copy()
    df[['pred_1d','pred_5d','pred_21d']] = preds
    return df

def backtest(df, top_k, hold_days):
    """
    df must have columns: start_date, ticker, pred_5d, ret_5d_f
    """
    # ensure sorted
    df = df.sort_values('start_date')
    pnl_records = []
    prev_positions = set()
    turnover_records = []
    
    all_dates = sorted(df['start_date'].unique())
    for dt in all_dates:
        sub = df[df['start_date'] == dt]
        # rank by prediction
        top = sub.nlargest(top_k, 'pred_5d')['ticker'].tolist()
        # compute turnover vs prev_positions
        if prev_positions:
            new = set(top)
            turnover = len(prev_positions.symmetric_difference(new)) / top_k
        else:
            turnover = 1.0
        turnover_records.append(turnover)
        prev_positions = set(top)
        
        # for each ticker, get its realized ret_5d_f
        rets = sub[sub['ticker'].isin(top)]['ret_5d_f']
        # equally weighted avg
        pnl = rets.mean()
        pnl_records.append({'date': dt, 'pnl': pnl})
    
    pnl_df = pd.DataFrame(pnl_records).set_index('date')
    turnover_df = pd.Series(turnover_records, index=all_dates, name='turnover')
    return pnl_df, turnover_df

def performance(pnl_ts, turnover_ts):
    # daily stats
    daily_ret = pnl_ts['pnl']
    mean = daily_ret.mean()
    std = daily_ret.std()
    sharpe = (mean / std) * np.sqrt(252/5)  # annualized assuming 5-day holding
    print(f"=== Strategy Performance ===")
    print(f"Days: {len(daily_ret)}")
    print(f"Avg 5-day return: {mean:.5f}")
    print(f"Std 5-day return: {std:.5f}")
    print(f"Annualized Sharpe: {sharpe:.2f}")
    print(f"Avg turnover per rebalance: {turnover_ts.mean():.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      type=str, default="data/kairos.duckdb")
    parser.add_argument("--window",  type=int, default=252, choices=[126,252])
    parser.add_argument("--model",   type=str, default="mh_lgbm.pkl")
    parser.add_argument("--top-k",   type=int, default=50,
                        help="Number of tickers to go long each rebalance")
    parser.add_argument("--hold",    type=int, default=5,
                        help="Holding period in days")
    args = parser.parse_args()

    # load
    meta, X = load_data(args.db, args.window)
    model = joblib.load(args.model)

    # predict
    df = predict(meta, X, model)

    # backtest
    pnl_ts, turnover_ts = backtest(df, top_k=args.top_k, hold_days=args.hold)

    # performance
    performance(pnl_ts, turnover_ts)
    # Save P&L series for plotting
    pnl_ts.to_csv("mh_strategy_pnl.csv")
    print("→ Saved P&L series to mh_strategy_pnl.csv")
    turnover_ts.to_csv("mh_strategy_turnover.csv")
    print("→ Saved turnover series to mh_strategy_turnover.csv")
if __name__ == "__main__":
    main()
