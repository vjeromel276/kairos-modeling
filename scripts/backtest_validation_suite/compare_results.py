#!/usr/bin/env python3
"""
compare_results.py

Compare results from multiple backtest validation runs.
Generates a side-by-side comparison report.

Usage:
    python compare_results.py results/run1 results/run2 results/run3

    Or compare all runs in a directory:
    python compare_results.py --all results/

Author: Kairos Quant Engineering
Version: 1.0
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def load_summary(result_dir: Path) -> dict:
    """Load summary.json from a result directory."""
    summary_path = result_dir / 'summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"No summary.json in {result_dir}")
    
    with open(summary_path, 'r') as f:
        return json.load(f)


def format_pct(value: float) -> str:
    """Format as percentage."""
    return f"{value:+.1%}" if value else "N/A"


def format_float(value: float, decimals: int = 2) -> str:
    """Format as float."""
    return f"{value:.{decimals}f}" if value else "N/A"


def compare_runs(result_dirs: list) -> pd.DataFrame:
    """Compare multiple runs and return comparison DataFrame."""
    
    rows = []
    
    for result_dir in result_dirs:
        result_dir = Path(result_dir)
        try:
            summary = load_summary(result_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue
        
        row = {
            'run_name': summary.get('config_name', result_dir.name),
            'run_dir': str(result_dir),
            'timestamp': summary.get('generated_at', 'N/A'),
            
            # Filters
            'adv_min': summary.get('filters', {}).get('adv_min', 0),
            'price_min': summary.get('filters', {}).get('price_min', 0),
            
            # Universe
            'avg_stocks': summary.get('universe', {}).get('avg_stocks', 0),
            
            # Validation - Decile
            'd10_return': summary.get('validation', {}).get('decile_10_return', 0),
            'd1_return': summary.get('validation', {}).get('decile_1_return', 0),
            'decile_spread': summary.get('validation', {}).get('decile_spread', 0),
            'monotonic': summary.get('validation', {}).get('decile_monotonic', False),
            
            # Validation - Quintile  
            'q1_return': summary.get('validation', {}).get('quintile_q1_return', 0),
            'q5_return': summary.get('validation', {}).get('quintile_q5_return', 0),
            'quintile_spread': summary.get('validation', {}).get('quintile_spread', 0),
            
            # Validation - IC
            'mean_ic': summary.get('validation', {}).get('mean_ic', 0),
            'ic_std': summary.get('validation', {}).get('ic_std', 0),
            'ic_pct_positive': summary.get('validation', {}).get('ic_pct_positive', 0),
            
            # Backtest - Portfolio
            'port_return': summary.get('backtest', {}).get('portfolio', {}).get('annual_return', 0),
            'port_vol': summary.get('backtest', {}).get('portfolio', {}).get('annual_vol', 0),
            'port_sharpe': summary.get('backtest', {}).get('portfolio', {}).get('sharpe', 0),
            'port_max_dd': summary.get('backtest', {}).get('portfolio', {}).get('max_drawdown', 0),
            
            # Backtest - Active
            'active_return': summary.get('backtest', {}).get('active', {}).get('annual_return', 0),
            'active_sharpe': summary.get('backtest', {}).get('active', {}).get('sharpe', 0),
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame):
    """Print formatted comparison table."""
    
    if df.empty:
        print("No results to compare.")
        return
    
    print("\n" + "=" * 100)
    print("BACKTEST VALIDATION COMPARISON")
    print("=" * 100)
    
    # Configuration section
    print("\n📋 CONFIGURATION")
    print("-" * 60)
    config_df = df[['run_name', 'adv_min', 'price_min', 'avg_stocks']].copy()
    config_df['adv_min'] = config_df['adv_min'].apply(lambda x: f"${x/1e6:.0f}M")
    config_df['price_min'] = config_df['price_min'].apply(lambda x: f"${x:.0f}")
    config_df['avg_stocks'] = config_df['avg_stocks'].apply(lambda x: f"{x:.0f}")
    config_df.columns = ['Config', 'ADV Min', 'Price Min', 'Avg Stocks']
    print(config_df.to_string(index=False))
    
    # Validation section - Decile
    print("\n📊 DECILE ANALYSIS")
    print("-" * 60)
    decile_df = df[['run_name', 'd10_return', 'd1_return', 'decile_spread', 'monotonic']].copy()
    decile_df['d10_return'] = decile_df['d10_return'].apply(format_pct)
    decile_df['d1_return'] = decile_df['d1_return'].apply(format_pct)
    decile_df['decile_spread'] = decile_df['decile_spread'].apply(format_pct)
    decile_df['monotonic'] = decile_df['monotonic'].apply(lambda x: '✓' if x else '✗')
    decile_df.columns = ['Config', 'D10 (Long)', 'D1 (Short)', 'Spread', 'Monotonic']
    print(decile_df.to_string(index=False))
    
    # Validation section - IC
    print("\n📈 IC ANALYSIS")
    print("-" * 60)
    ic_df = df[['run_name', 'mean_ic', 'ic_std', 'ic_pct_positive']].copy()
    ic_df['mean_ic'] = ic_df['mean_ic'].apply(lambda x: f"{x:.4f}")
    ic_df['ic_std'] = ic_df['ic_std'].apply(lambda x: f"{x:.4f}")
    ic_df['ic_pct_positive'] = ic_df['ic_pct_positive'].apply(lambda x: f"{x:.0%}")
    ic_df.columns = ['Config', 'Mean IC', 'IC Std', '% Positive']
    print(ic_df.to_string(index=False))
    
    # Backtest section
    print("\n🎯 BACKTEST RESULTS (Risk4 Portfolio)")
    print("-" * 60)
    bt_df = df[['run_name', 'port_return', 'port_vol', 'port_sharpe', 'port_max_dd']].copy()
    bt_df['port_return'] = bt_df['port_return'].apply(format_pct)
    bt_df['port_vol'] = bt_df['port_vol'].apply(lambda x: f"{x:.1%}")
    bt_df['port_sharpe'] = bt_df['port_sharpe'].apply(lambda x: f"{x:.2f}")
    bt_df['port_max_dd'] = bt_df['port_max_dd'].apply(lambda x: f"{x:.1%}")
    bt_df.columns = ['Config', 'Ann Return', 'Ann Vol', 'Sharpe', 'Max DD']
    print(bt_df.to_string(index=False))
    
    # Active returns
    print("\n💹 ACTIVE PERFORMANCE (vs Benchmark)")
    print("-" * 60)
    active_df = df[['run_name', 'active_return', 'active_sharpe']].copy()
    active_df['active_return'] = active_df['active_return'].apply(format_pct)
    active_df['active_sharpe'] = active_df['active_sharpe'].apply(lambda x: f"{x:.2f}")
    active_df.columns = ['Config', 'Active Return', 'Active Sharpe']
    print(active_df.to_string(index=False))
    
    # Winner determination
    print("\n🏆 BEST CONFIGURATION")
    print("-" * 60)
    
    # Best by different metrics
    best_sharpe_idx = df['port_sharpe'].idxmax()
    best_active_idx = df['active_sharpe'].idxmax()
    best_ic_idx = df['mean_ic'].idxmax()
    best_spread_idx = df['decile_spread'].idxmax()
    
    print(f"  Best Portfolio Sharpe:  {df.loc[best_sharpe_idx, 'run_name']} ({df.loc[best_sharpe_idx, 'port_sharpe']:.2f})")
    print(f"  Best Active Sharpe:     {df.loc[best_active_idx, 'run_name']} ({df.loc[best_active_idx, 'active_sharpe']:.2f})")
    print(f"  Best Mean IC:           {df.loc[best_ic_idx, 'run_name']} ({df.loc[best_ic_idx, 'mean_ic']:.4f})")
    print(f"  Best Decile Spread:     {df.loc[best_spread_idx, 'run_name']} ({df.loc[best_spread_idx, 'decile_spread']:.1%})")
    
    print("\n" + "=" * 100)


def save_comparison(df: pd.DataFrame, output_path: str):
    """Save comparison to CSV."""
    df.to_csv(output_path, index=False)
    print(f"\nComparison saved to: {output_path}")


def find_all_runs(base_dir: str) -> list:
    """Find all result directories in base directory."""
    base_path = Path(base_dir)
    runs = []
    
    for item in base_path.iterdir():
        if item.is_dir() and (item / 'summary.json').exists():
            runs.append(item)
    
    # Sort by timestamp (directory name starts with timestamp)
    runs.sort(key=lambda x: x.name)
    
    return runs


def main():
    parser = argparse.ArgumentParser(description='Compare backtest validation results')
    parser.add_argument('result_dirs', nargs='*', help='Result directories to compare')
    parser.add_argument('--all', type=str, help='Compare all runs in this directory')
    parser.add_argument('--output', '-o', type=str, help='Save comparison to CSV')
    args = parser.parse_args()
    
    # Get result directories
    if args.all:
        result_dirs = find_all_runs(args.all)
        if not result_dirs:
            print(f"No result directories found in {args.all}")
            return 1
    elif args.result_dirs:
        result_dirs = args.result_dirs
    else:
        print("Specify result directories or use --all")
        return 1
    
    print(f"Comparing {len(result_dirs)} runs...")
    
    # Compare
    df = compare_runs(result_dirs)
    
    # Print
    print_comparison(df)
    
    # Save if requested
    if args.output:
        save_comparison(df, args.output)
    
    return 0


if __name__ == "__main__":
    exit(main())