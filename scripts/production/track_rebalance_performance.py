#!/usr/bin/env python3
"""
track_rebalance_performance.py
==============================
Track portfolio performance between rebalance dates.

Compares picks from consecutive rebalance periods to calculate:
- Individual stock returns (weighted and unweighted)
- Portfolio return vs benchmark
- Winners/losers analysis
- Sector attribution

Usage:
    # Compare two specific rebalance dates
    python track_rebalance_performance.py \
        --picks-dir outputs/rebalance \
        --from-date 2025-12-19 \
        --to-date 2025-12-26
    
    # Track all rebalances in directory
    python track_rebalance_performance.py \
        --picks-dir outputs/rebalance \
        --all
    
    # Fetch live prices for current holdings
    python track_rebalance_performance.py \
        --picks-dir outputs/rebalance \
        --from-date 2025-12-26 \
        --live

Output:
    - Console summary
    - CSV with detailed stock-level returns
    - Cumulative performance log (appended)
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def get_alpaca_api():
    """Get Alpaca API connection."""
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(
        os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP"),
        os.getenv("ALPACA_SECRET_KEY", "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB"),
        os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    )


def get_alpaca_prices(symbols: List[str]) -> Dict[str, float]:
    """Fetch current prices from Alpaca."""
    try:
        api = get_alpaca_api()
        
        prices = {}
        # Alpaca allows up to 200 symbols per request
        for i in range(0, len(symbols), 200):
            batch = symbols[i:i+200]
            try:
                snapshots = api.get_snapshots(batch)
                for symbol, snapshot in snapshots.items():
                    if snapshot and snapshot.latest_trade:
                        prices[symbol] = snapshot.latest_trade.price
            except Exception as e:
                print(f"  Warning: Could not fetch batch {i}: {e}")
        
        return prices
    except ImportError:
        print("Warning: alpaca_trade_api not installed, cannot fetch live prices")
        return {}


def get_benchmark_return(from_date: str, to_date: str = None, benchmark: str = "SPY") -> Optional[float]:
    """
    Get benchmark return between two dates.
    
    Args:
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD) or None for live
        benchmark: Benchmark symbol (default SPY)
    
    Returns:
        Return as decimal (e.g., 0.02 for 2%) or None if unavailable
    """
    try:
        api = get_alpaca_api()
        
        # Get historical bars for benchmark
        from datetime import datetime, timedelta
        
        start_dt = datetime.strptime(from_date, "%Y-%m-%d")
        
        if to_date and to_date != "LIVE":
            end_dt = datetime.strptime(to_date, "%Y-%m-%d")
        else:
            end_dt = datetime.now()
        
        # Fetch daily bars
        bars = api.get_bars(
            benchmark,
            "1Day",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            limit=30,
        ).df
        
        if len(bars) < 2:
            print(f"  Warning: Not enough benchmark data for {benchmark}")
            return None
        
        # Get first and last close prices
        start_price = bars.iloc[0]["close"]
        end_price = bars.iloc[-1]["close"]
        
        benchmark_return = (end_price / start_price) - 1
        return benchmark_return
        
    except Exception as e:
        print(f"  Warning: Could not fetch benchmark {benchmark}: {e}")
        return None


def load_picks(picks_path: str) -> pd.DataFrame:
    """Load picks.csv file."""
    df = pd.read_csv(picks_path)
    return df


def find_rebalance_dates(picks_dir: str) -> List[str]:
    """Find all rebalance dates in the picks directory."""
    dates = []
    for item in os.listdir(picks_dir):
        item_path = os.path.join(picks_dir, item)
        if os.path.isdir(item_path):
            picks_file = os.path.join(item_path, "picks.csv")
            if os.path.exists(picks_file):
                dates.append(item)
    return sorted(dates)


def get_historical_prices(symbols: List[str], date: str) -> Dict[str, float]:
    """Fetch historical closing prices for a specific date from Alpaca."""
    if not symbols:
        return {}
    
    try:
        api = get_alpaca_api()
        from datetime import datetime, timedelta
        
        # Parse date and get a range around it
        target_dt = datetime.strptime(date, "%Y-%m-%d")
        start_dt = target_dt - timedelta(days=5)  # Look back a few days
        end_dt = target_dt + timedelta(days=1)
        
        prices = {}
        # Fetch in batches
        for i in range(0, len(symbols), 200):
            batch = symbols[i:i+200]
            try:
                bars = api.get_bars(
                    batch,
                    "1Day",
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                )
                
                # Group by symbol and get the latest bar before/on target date
                for bar in bars:
                    symbol = bar.S
                    bar_date = bar.t.date()
                    if bar_date <= target_dt.date():
                        # Keep the latest price up to target date
                        if symbol not in prices or bar_date > prices[symbol][0]:
                            prices[symbol] = (bar_date, bar.c)
                
            except Exception as e:
                print(f"  Warning: Could not fetch historical batch {i}: {e}")
        
        # Return just the prices, not the dates
        return {symbol: price for symbol, (_, price) in prices.items()}
        
    except Exception as e:
        print(f"  Warning: Could not fetch historical prices: {e}")
        return {}


def calculate_returns(
    picks_start: pd.DataFrame,
    picks_end: pd.DataFrame = None,
    end_prices: Dict[str, float] = None,
    end_date: str = None,
) -> pd.DataFrame:
    """
    Calculate returns for each stock in the portfolio.
    
    Args:
        picks_start: Starting picks with entry prices
        picks_end: Ending picks with end prices (optional)
        end_prices: Dict of symbol -> price for end (alternative to picks_end)
        end_date: Date string for fetching dropped stock prices (YYYY-MM-DD)
    
    Returns:
        DataFrame with returns for each stock
    """
    results = []
    dropped_tickers = []
    
    for _, row in picks_start.iterrows():
        ticker = row["ticker"]
        weight = row["weight"]
        entry_price = row["price"]
        sector = row.get("sector", "Unknown")
        alpha_z = row.get("alpha_z", 0)
        
        # Get end price
        end_price = None
        still_held = False
        
        if end_prices and ticker in end_prices:
            end_price = end_prices[ticker]
            still_held = True  # If we're checking live prices, position is still held
        elif picks_end is not None:
            end_row = picks_end[picks_end["ticker"] == ticker]
            if len(end_row) > 0:
                end_price = end_row.iloc[0]["price"]
                still_held = True
            else:
                # Stock was dropped - mark for historical price fetch
                dropped_tickers.append(ticker)
        
        results.append({
            "ticker": ticker,
            "sector": sector,
            "weight": weight,
            "alpha_z": alpha_z,
            "entry_price": entry_price,
            "end_price": end_price,
            "still_held": still_held,
        })
    
    # Fetch historical prices for dropped stocks
    if dropped_tickers and end_date and end_date != "LIVE":
        print(f"  Fetching historical prices for {len(dropped_tickers)} dropped stocks...")
        historical_prices = get_historical_prices(dropped_tickers, end_date)
        
        # Update results with fetched prices
        for result in results:
            if result["ticker"] in historical_prices:
                result["end_price"] = historical_prices[result["ticker"]]
    
    # Calculate returns
    for result in results:
        entry_price = result["entry_price"]
        end_price = result["end_price"]
        weight = result["weight"]
        
        if end_price is not None and entry_price > 0:
            stock_return = (end_price / entry_price) - 1
            weighted_return = stock_return * weight
            result["return_pct"] = stock_return * 100
            result["weighted_return_pct"] = weighted_return * 100
        else:
            result["return_pct"] = None
            result["weighted_return_pct"] = None
    
    return pd.DataFrame(results)


def print_performance_summary(
    returns_df: pd.DataFrame,
    from_date: str,
    to_date: str,
    benchmark_return: float = None,
):
    """Print formatted performance summary."""
    
    valid_returns = returns_df.dropna(subset=["return_pct"]).copy()
    
    if len(valid_returns) == 0:
        print("No valid returns to calculate.")
        return
    
    # Portfolio return (weighted)
    portfolio_return = valid_returns["weighted_return_pct"].sum()
    
    # Equal-weight return (for comparison)
    equal_weight_return = valid_returns["return_pct"].mean()
    
    # Winners/losers
    winners = valid_returns[valid_returns["return_pct"] > 0]
    losers = valid_returns[valid_returns["return_pct"] < 0]
    
    print("\n" + "=" * 70)
    print(f"REBALANCE PERFORMANCE: {from_date} → {to_date}")
    print("=" * 70)
    
    print(f"\n--- PORTFOLIO VS BENCHMARK ---")
    print(f"Portfolio Return (weighted):    {portfolio_return:>8.2f}%")
    if benchmark_return is not None:
        bench_pct = benchmark_return * 100
        active_return = portfolio_return - bench_pct
        print(f"Benchmark Return (SPY):         {bench_pct:>8.2f}%")
        print(f"Active Return:                  {active_return:>8.2f}%")
        if active_return > 0:
            print(f"  ✓ OUTPERFORMED benchmark by {active_return:.2f}%")
        else:
            print(f"  ✗ UNDERPERFORMED benchmark by {abs(active_return):.2f}%")
    else:
        print(f"Benchmark Return:               (unavailable)")
    print(f"Equal-Weight Return:            {equal_weight_return:>8.2f}%")
    
    print(f"\n--- WINNERS / LOSERS ---")
    print(f"Winners: {len(winners)} ({len(winners)/len(valid_returns)*100:.0f}%)")
    print(f"Losers:  {len(losers)} ({len(losers)/len(valid_returns)*100:.0f}%)")
    print(f"Win Rate: {len(winners)/(len(winners)+len(losers))*100:.1f}%")
    
    # Top winners
    print(f"\n--- TOP 5 WINNERS ---")
    top_winners = valid_returns.nlargest(5, "return_pct")
    for _, row in top_winners.iterrows():
        print(f"  {row['ticker']:<6} {row['sector']:<20} "
              f"Return: {row['return_pct']:>7.2f}%  "
              f"Weight: {row['weight']*100:>5.2f}%  "
              f"Contribution: {row['weighted_return_pct']:>6.2f}%")
    
    # Top losers
    print(f"\n--- TOP 5 LOSERS ---")
    top_losers = valid_returns.nsmallest(5, "return_pct")
    for _, row in top_losers.iterrows():
        print(f"  {row['ticker']:<6} {row['sector']:<20} "
              f"Return: {row['return_pct']:>7.2f}%  "
              f"Weight: {row['weight']*100:>5.2f}%  "
              f"Contribution: {row['weighted_return_pct']:>6.2f}%")
    
    # Sector attribution
    print(f"\n--- SECTOR ATTRIBUTION ---")
    sector_perf = valid_returns.groupby("sector", observed=True).agg({
        "weighted_return_pct": "sum",
        "return_pct": "mean",
        "weight": "sum",
        "ticker": "count",
    }).rename(columns={"ticker": "count"})
    sector_perf = sector_perf.sort_values("weighted_return_pct", ascending=False)
    
    for sector, row in sector_perf.iterrows():
        print(f"  {sector:<25} "
              f"Contrib: {row['weighted_return_pct']:>6.2f}%  "
              f"Avg Ret: {row['return_pct']:>6.2f}%  "
              f"Weight: {row['weight']*100:>5.1f}%  "
              f"({int(row['count'])} stocks)")
    
    # Alpha analysis
    print(f"\n--- ALPHA SIGNAL ANALYSIS ---")
    # Group by alpha quartile
    valid_returns["alpha_quartile"] = pd.qcut(
        valid_returns["alpha_z"], 
        q=4, 
        labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
    )
    alpha_perf = valid_returns.groupby("alpha_quartile", observed=True).agg({
        "return_pct": "mean",
        "ticker": "count",
    }).rename(columns={"ticker": "count"})
    
    for quartile, row in alpha_perf.iterrows():
        print(f"  {quartile:<12} Avg Return: {row['return_pct']:>7.2f}%  ({int(row['count'])} stocks)")
    
    # Check if high alpha outperformed low alpha
    q4_return = alpha_perf.loc["Q4 (high)", "return_pct"]
    q1_return = alpha_perf.loc["Q1 (low)", "return_pct"]
    spread = q4_return - q1_return
    print(f"\n  Alpha Spread (Q4 - Q1): {spread:>7.2f}%")
    if spread > 0:
        print(f"  ✓ High alpha outperformed low alpha")
    else:
        print(f"  ✗ High alpha underperformed low alpha")
    
    print("\n" + "=" * 70)


def save_performance_log(
    returns_df: pd.DataFrame,
    from_date: str,
    to_date: str,
    output_dir: str,
    benchmark_return: float = None,
):
    """Save detailed performance to CSV and append to cumulative log."""
    
    # Save detailed returns for this period
    detail_file = os.path.join(output_dir, f"returns_{from_date}_to_{to_date}.csv")
    returns_df.to_csv(detail_file, index=False)
    print(f"\nDetailed returns saved to: {detail_file}")
    
    # Append summary to cumulative log
    valid_returns = returns_df.dropna(subset=["return_pct"])
    if len(valid_returns) == 0:
        return
    
    portfolio_return = valid_returns["weighted_return_pct"].sum()
    equal_weight_return = valid_returns["return_pct"].mean()
    winners = len(valid_returns[valid_returns["return_pct"] > 0])
    total = len(valid_returns)
    
    summary = {
        "from_date": from_date,
        "to_date": to_date,
        "portfolio_return_pct": portfolio_return,
        "benchmark_return_pct": benchmark_return * 100 if benchmark_return is not None else None,
        "active_return_pct": portfolio_return - (benchmark_return * 100) if benchmark_return is not None else None,
        "equal_weight_return_pct": equal_weight_return,
        "win_rate_pct": winners / total * 100,
        "n_positions": total,
        "timestamp": datetime.now().isoformat(),
    }
    
    log_file = os.path.join(output_dir, "performance_log.csv")
    
    if os.path.exists(log_file):
        log_df = pd.read_csv(log_file)
        # Check if this period already logged
        exists = ((log_df["from_date"] == from_date) & 
                  (log_df["to_date"] == to_date)).any()
        if not exists:
            log_df = pd.concat([log_df, pd.DataFrame([summary])], ignore_index=True)
    else:
        log_df = pd.DataFrame([summary])
    
    log_df.to_csv(log_file, index=False)
    print(f"Performance log updated: {log_file}")
    
    # Print cumulative performance
    if len(log_df) > 1:
        cumulative = (1 + log_df["portfolio_return_pct"] / 100).prod() - 1
        print(f"\nCumulative return across {len(log_df)} periods: {cumulative*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Track Kairos rebalance performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--picks-dir", default="outputs/rebalance",
                        help="Directory containing rebalance outputs")
    parser.add_argument("--from-date", help="Starting rebalance date (YYYY-MM-DD)")
    parser.add_argument("--to-date", help="Ending rebalance date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", 
                        help="Process all consecutive rebalance periods")
    parser.add_argument("--live", action="store_true",
                        help="Fetch live prices for current holdings")
    parser.add_argument("--output-dir", default="outputs/performance",
                        help="Directory for output files")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find available rebalance dates
    available_dates = find_rebalance_dates(args.picks_dir)
    print(f"Found {len(available_dates)} rebalance dates: {available_dates}")
    
    if args.all:
        # Process all consecutive periods
        if len(available_dates) < 2:
            print("Need at least 2 rebalance dates to compare")
            sys.exit(1)
        
        for i in range(len(available_dates) - 1):
            from_date = available_dates[i]
            to_date = available_dates[i + 1]
            
            picks_start = load_picks(
                os.path.join(args.picks_dir, from_date, "picks.csv")
            )
            picks_end = load_picks(
                os.path.join(args.picks_dir, to_date, "picks.csv")
            )
            
            # Fetch benchmark return
            benchmark_return = get_benchmark_return(from_date, to_date)
            
            returns_df = calculate_returns(picks_start, picks_end, end_date=to_date)
            print_performance_summary(returns_df, from_date, to_date, benchmark_return)
            save_performance_log(returns_df, from_date, to_date, args.output_dir, benchmark_return)
    
    elif args.live:
        # Compare from_date to current live prices
        if not args.from_date:
            args.from_date = available_dates[-1]  # Use most recent
        
        picks_path = os.path.join(args.picks_dir, args.from_date, "picks.csv")
        if not os.path.exists(picks_path):
            print(f"Error: No picks found for {args.from_date}")
            sys.exit(1)
        
        picks_start = load_picks(picks_path)
        symbols = picks_start["ticker"].tolist()
        
        print(f"Fetching live prices for {len(symbols)} symbols...")
        live_prices = get_alpaca_prices(symbols)
        print(f"Got prices for {len(live_prices)} symbols")
        
        # Fetch benchmark return
        benchmark_return = get_benchmark_return(args.from_date, None)
        
        returns_df = calculate_returns(picks_start, end_prices=live_prices)
        print_performance_summary(returns_df, args.from_date, "LIVE", benchmark_return)
        save_performance_log(returns_df, args.from_date, "LIVE", args.output_dir, benchmark_return)
    
    else:
        # Compare two specific dates
        if not args.from_date or not args.to_date:
            print("Error: Must specify --from-date and --to-date, or use --all or --live")
            sys.exit(1)
        
        picks_start_path = os.path.join(args.picks_dir, args.from_date, "picks.csv")
        picks_end_path = os.path.join(args.picks_dir, args.to_date, "picks.csv")
        
        if not os.path.exists(picks_start_path):
            print(f"Error: No picks found for {args.from_date}")
            sys.exit(1)
        if not os.path.exists(picks_end_path):
            print(f"Error: No picks found for {args.to_date}")
            sys.exit(1)
        
        picks_start = load_picks(picks_start_path)
        picks_end = load_picks(picks_end_path)
        
        # Fetch benchmark return
        benchmark_return = get_benchmark_return(args.from_date, args.to_date)
        
        returns_df = calculate_returns(picks_start, picks_end, end_date=args.to_date)
        print_performance_summary(returns_df, args.from_date, args.to_date, benchmark_return)
        save_performance_log(returns_df, args.from_date, args.to_date, args.output_dir, benchmark_return)


if __name__ == "__main__":
    main()