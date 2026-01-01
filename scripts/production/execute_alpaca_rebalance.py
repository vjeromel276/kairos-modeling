#!/usr/bin/env python3
"""
execute_alpaca_rebalance.py
===========================
Execute Kairos rebalance picks via Alpaca API.

Modes:
  1. Initial portfolio (no current positions) - buys all picks
  2. Rebalance (existing positions) - uses trades.csv for delta

Usage:
    # Preview what would happen (no orders submitted)
    python execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --preview
    
    # Execute trades
    python execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute
    
    # Execute with custom portfolio value (default uses account equity)
    python execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --portfolio-value 100000 --execute

Environment:
    ALPACA_API_KEY: Your Alpaca API key
    ALPACA_SECRET_KEY: Your Alpaca secret key
    ALPACA_BASE_URL: API base URL (default: paper trading)
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# Alpaca configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_api():
    """Initialize Alpaca API connection."""
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


def get_account_info(api) -> Dict:
    """Get account status and buying power."""
    account = api.get_account()
    return {
        "status": account.status,
        "cash": float(account.cash),
        "portfolio_value": float(account.portfolio_value),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
    }


def get_current_positions(api) -> Dict[str, Dict]:
    """Get current portfolio positions."""
    positions = api.list_positions()
    return {
        p.symbol: {
            "qty": int(p.qty),
            "market_value": float(p.market_value),
            "current_price": float(p.current_price),
            "avg_entry_price": float(p.avg_entry_price),
        }
        for p in positions
    }


def load_picks(picks_path: str, portfolio_value: float = None) -> pd.DataFrame:
    """Load picks.csv and optionally rescale for portfolio value."""
    df = pd.read_csv(picks_path)
    
    # If portfolio value specified, recalculate shares based on weights
    if portfolio_value:
        df["target_value"] = df["weight"] * portfolio_value
        # Recalculate shares (need current prices - use price from file as estimate)
        df["shares"] = (df["target_value"] / df["price"]).astype(int)
    
    return df


def calculate_orders(
    picks: pd.DataFrame, 
    current_positions: Dict[str, Dict],
    portfolio_value: float
) -> List[Dict]:
    """
    Calculate orders needed to rebalance to target.
    
    Returns list of orders: [{symbol, side, qty, notional_value}, ...]
    """
    orders = []
    
    # Build target position map
    target_positions = {
        row["ticker"]: {
            "shares": int(row["shares"]),
            "target_value": row["target_value"],
            "weight": row["weight"],
        }
        for _, row in picks.iterrows()
    }
    
    # Calculate sells first (positions we need to reduce or exit)
    for symbol, pos in current_positions.items():
        current_qty = pos["qty"]
        target_qty = target_positions.get(symbol, {}).get("shares", 0)
        
        if current_qty > target_qty:
            sell_qty = current_qty - target_qty
            orders.append({
                "symbol": symbol,
                "side": "sell",
                "qty": sell_qty,
                "notional_value": sell_qty * pos["current_price"],
                "reason": "exit" if target_qty == 0 else "reduce",
            })
    
    # Calculate buys (new positions or increases)
    for symbol, target in target_positions.items():
        current_qty = current_positions.get(symbol, {}).get("qty", 0)
        target_qty = target["shares"]
        
        if target_qty > current_qty:
            buy_qty = target_qty - current_qty
            # Estimate price from picks
            pick_row = picks[picks["ticker"] == symbol].iloc[0]
            est_price = pick_row["price"]
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "qty": buy_qty,
                "notional_value": buy_qty * est_price,
                "reason": "new" if current_qty == 0 else "increase",
            })
    
    # Sort: sells first, then buys by notional value descending
    sells = sorted([o for o in orders if o["side"] == "sell"], 
                   key=lambda x: -x["notional_value"])
    buys = sorted([o for o in orders if o["side"] == "buy"], 
                  key=lambda x: -x["notional_value"])
    
    return sells + buys


def preview_orders(orders: List[Dict], account_info: Dict):
    """Print preview of orders without executing."""
    print("\n" + "=" * 70)
    print("ORDER PREVIEW (no orders will be submitted)")
    print("=" * 70)
    
    print(f"\nAccount Status: {account_info['status']}")
    print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
    print(f"Cash: ${account_info['cash']:,.2f}")
    print(f"Buying Power: ${account_info['buying_power']:,.2f}")
    
    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]
    
    total_sell = sum(o["notional_value"] for o in sells)
    total_buy = sum(o["notional_value"] for o in buys)
    
    print(f"\n--- SELLS ({len(sells)} orders, ${total_sell:,.2f} total) ---")
    for o in sells[:10]:  # Show top 10
        print(f"  SELL {o['qty']:>6} {o['symbol']:<6} ~${o['notional_value']:>10,.2f} ({o['reason']})")
    if len(sells) > 10:
        print(f"  ... and {len(sells) - 10} more sells")
    
    print(f"\n--- BUYS ({len(buys)} orders, ${total_buy:,.2f} total) ---")
    for o in buys[:15]:  # Show top 15
        print(f"  BUY  {o['qty']:>6} {o['symbol']:<6} ~${o['notional_value']:>10,.2f} ({o['reason']})")
    if len(buys) > 15:
        print(f"  ... and {len(buys) - 15} more buys")
    
    print(f"\n--- SUMMARY ---")
    print(f"Total Sells: ${total_sell:,.2f}")
    print(f"Total Buys:  ${total_buy:,.2f}")
    print(f"Net:         ${total_buy - total_sell:,.2f}")
    
    if total_buy > account_info["buying_power"]:
        print(f"\n⚠️  WARNING: Total buys (${total_buy:,.2f}) exceed buying power (${account_info['buying_power']:,.2f})")
        print("    Orders may be rejected or partially filled.")
    
    print("\n" + "=" * 70)
    print("To execute these orders, run with --execute flag")
    print("=" * 70 + "\n")


def execute_orders(api, orders: List[Dict], dry_run: bool = False) -> List[Dict]:
    """
    Submit orders to Alpaca.
    
    Returns list of order results.
    """
    results = []
    
    print("\n" + "=" * 70)
    print("EXECUTING ORDERS")
    print("=" * 70 + "\n")
    
    for i, order in enumerate(orders, 1):
        symbol = order["symbol"]
        side = order["side"]
        qty = order["qty"]
        
        if qty <= 0:
            continue
        
        print(f"[{i}/{len(orders)}] {side.upper()} {qty} {symbol}...", end=" ")
        
        if dry_run:
            print("(dry run - skipped)")
            results.append({"symbol": symbol, "status": "dry_run"})
            continue
        
        try:
            # Submit market order
            submitted = api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type="market",
                time_in_force="day",  # Good for the day
            )
            print(f"✓ Order ID: {submitted.id}")
            results.append({
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_id": submitted.id,
                "status": "submitted",
            })
        except Exception as e:
            print(f"✗ FAILED: {e}")
            results.append({
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "status": "failed",
                "error": str(e),
            })
    
    # Summary
    submitted = len([r for r in results if r["status"] == "submitted"])
    failed = len([r for r in results if r["status"] == "failed"])
    
    print(f"\n--- EXECUTION SUMMARY ---")
    print(f"Submitted: {submitted}")
    print(f"Failed: {failed}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Execute Kairos rebalance via Alpaca",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument("--picks", required=True, help="Path to picks.csv")
    parser.add_argument("--trades", help="Path to trades.csv (optional, for delta rebalance)")
    parser.add_argument("--portfolio-value", type=float, help="Override portfolio value for share calculation")
    parser.add_argument("--preview", action="store_true", help="Preview orders without executing")
    parser.add_argument("--execute", action="store_true", help="Actually submit orders")
    parser.add_argument("--dry-run", action="store_true", help="Go through motions but don't submit")
    
    args = parser.parse_args()
    
    if not args.preview and not args.execute:
        print("Error: Must specify --preview or --execute")
        sys.exit(1)
    
    # Connect to Alpaca
    print("Connecting to Alpaca...")
    api = get_api()
    
    # Get account info
    account_info = get_account_info(api)
    print(f"Account status: {account_info['status']}")
    
    if account_info["status"] != "ACTIVE":
        print(f"Error: Account not active (status: {account_info['status']})")
        sys.exit(1)
    
    # Get current positions
    current_positions = get_current_positions(api)
    print(f"Current positions: {len(current_positions)}")
    
    # Determine portfolio value
    portfolio_value = args.portfolio_value or account_info["equity"]
    print(f"Portfolio value for calculation: ${portfolio_value:,.2f}")
    
    # Load picks
    picks = load_picks(args.picks, portfolio_value if args.portfolio_value else None)
    print(f"Loaded {len(picks)} picks from {args.picks}")
    
    # Calculate orders
    orders = calculate_orders(picks, current_positions, portfolio_value)
    print(f"Calculated {len(orders)} orders")
    
    if args.preview:
        preview_orders(orders, account_info)
    elif args.execute:
        # Confirm before executing
        print(f"\n⚠️  About to submit {len(orders)} orders to Alpaca")
        print(f"    Account: {ALPACA_BASE_URL}")
        confirm = input("    Type 'YES' to confirm: ")
        
        if confirm != "YES":
            print("Cancelled.")
            sys.exit(0)
        
        results = execute_orders(api, orders, dry_run=args.dry_run)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"alpaca_orders_{timestamp}.csv"
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"\nOrder results saved to: {results_file}")


if __name__ == "__main__":
    main()