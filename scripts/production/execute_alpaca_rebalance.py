#!/usr/bin/env python3
"""
execute_alpaca_rebalance.py
===========================
Execute Kairos rebalance picks via Alpaca API.

Modes:
  1. Initial portfolio (no current positions) - buys all picks
  2. Rebalance (existing positions) - calculates delta from Alpaca positions

Order Types:
  - Default: Market orders (immediate fill during market hours)
  - --moo: Market-on-Open orders (queue for next market open)
  - --loo: Limit-on-Open orders with 1% buffer (recommended for weekly rebalance)
           Buy limit = Friday close × 1.01
           Sell limit = Friday close × 0.99

Usage:
    # Preview what would happen (no orders submitted)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --preview
    
    # Execute trades immediately (market hours)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute
    
    # Execute as Market-on-Open (submit Friday night, fills Monday 9:30 AM)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute --moo
    
    # Execute as Limit-on-Open with 1% slippage protection (RECOMMENDED)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute --loo
    
    # Execute with custom portfolio value (default uses account equity)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --portfolio-value 100000 --execute --loo

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

# LOO slippage buffer (1% = 0.01)
LOO_SLIPPAGE_PCT = 0.01


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


def format_limit_price(price: float) -> str:
    """
    Format limit price according to Alpaca sub-penny rules.
    - Price >= $1.00: max 2 decimal places
    - Price < $1.00: max 4 decimal places
    """
    if price >= 1.0:
        return f"{price:.2f}"
    else:
        return f"{price:.4f}"


def calculate_limit_price(ref_price: float, side: str) -> float:
    """
    Calculate limit price with slippage buffer.
    - Buy: ref_price × 1.01 (willing to pay up to 1% more)
    - Sell: ref_price × 0.99 (willing to accept up to 1% less)
    """
    if side == "buy":
        return ref_price * (1 + LOO_SLIPPAGE_PCT)
    else:  # sell
        return ref_price * (1 - LOO_SLIPPAGE_PCT)


def calculate_orders(
    picks: pd.DataFrame, 
    current_positions: Dict[str, Dict],
    portfolio_value: float
) -> List[Dict]:
    """
    Calculate orders needed to rebalance to target.
    
    Returns list of orders: [{symbol, side, qty, notional_value, ref_price}, ...]
    """
    orders = []
    
    # Build target position map with reference prices
    target_positions = {
        row["ticker"]: {
            "shares": int(row["shares"]),
            "target_value": row["target_value"],
            "weight": row["weight"],
            "ref_price": row["price"],  # Friday close price for LOO
        }
        for _, row in picks.iterrows()
    }
    
    # Calculate sells first (positions we need to reduce or exit)
    for symbol, pos in current_positions.items():
        current_qty = pos["qty"]
        target = target_positions.get(symbol, {})
        target_qty = target.get("shares", 0)
        
        if current_qty > target_qty:
            sell_qty = current_qty - target_qty
            # Use current price for sells of positions not in picks
            ref_price = target.get("ref_price", pos["current_price"])
            orders.append({
                "symbol": symbol,
                "side": "sell",
                "qty": sell_qty,
                "notional_value": sell_qty * pos["current_price"],
                "ref_price": ref_price,
                "reason": "exit" if target_qty == 0 else "reduce",
            })
    
    # Calculate buys (new positions or increases)
    for symbol, target in target_positions.items():
        current_qty = current_positions.get(symbol, {}).get("qty", 0)
        target_qty = target["shares"]
        
        if target_qty > current_qty:
            buy_qty = target_qty - current_qty
            ref_price = target["ref_price"]
            orders.append({
                "symbol": symbol,
                "side": "buy",
                "qty": buy_qty,
                "notional_value": buy_qty * ref_price,
                "ref_price": ref_price,
                "reason": "new" if current_qty == 0 else "increase",
            })
    
    # Sort: sells first, then buys by notional value descending
    sells = sorted([o for o in orders if o["side"] == "sell"], 
                   key=lambda x: -x["notional_value"])
    buys = sorted([o for o in orders if o["side"] == "buy"], 
                  key=lambda x: -x["notional_value"])
    
    return sells + buys


def preview_orders(orders: List[Dict], account_info: Dict, use_moo: bool = False, use_loo: bool = False):
    """Print preview of orders without executing."""
    print("\n" + "=" * 70)
    print("ORDER PREVIEW (no orders will be submitted)")
    print("=" * 70)
    
    if use_loo:
        order_type = f"Limit-on-Open (LOO) with {LOO_SLIPPAGE_PCT*100:.0f}% buffer"
    elif use_moo:
        order_type = "Market-on-Open (MOO)"
    else:
        order_type = "Market (immediate)"
    
    print(f"\nOrder Type: {order_type}")
    print(f"Account Status: {account_info['status']}")
    print(f"Portfolio Value: ${account_info['portfolio_value']:,.2f}")
    print(f"Cash: ${account_info['cash']:,.2f}")
    print(f"Buying Power: ${account_info['buying_power']:,.2f}")
    
    sells = [o for o in orders if o["side"] == "sell"]
    buys = [o for o in orders if o["side"] == "buy"]
    
    total_sell = sum(o["notional_value"] for o in sells)
    total_buy = sum(o["notional_value"] for o in buys)
    
    print(f"\n--- SELLS ({len(sells)} orders, ${total_sell:,.2f} total) ---")
    for o in sells[:10]:  # Show top 10
        if use_loo:
            limit = calculate_limit_price(o["ref_price"], "sell")
            print(f"  SELL {o['qty']:>6} {o['symbol']:<6} ~${o['notional_value']:>10,.2f} limit ${limit:>8.2f} ({o['reason']})")
        else:
            print(f"  SELL {o['qty']:>6} {o['symbol']:<6} ~${o['notional_value']:>10,.2f} ({o['reason']})")
    if len(sells) > 10:
        print(f"  ... and {len(sells) - 10} more sells")
    
    print(f"\n--- BUYS ({len(buys)} orders, ${total_buy:,.2f} total) ---")
    for o in buys[:15]:  # Show top 15
        if use_loo:
            limit = calculate_limit_price(o["ref_price"], "buy")
            print(f"  BUY  {o['qty']:>6} {o['symbol']:<6} ~${o['notional_value']:>10,.2f} limit ${limit:>8.2f} ({o['reason']})")
        else:
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
    
    if use_loo:
        print(f"\n📅 LOO orders will execute at next market open (9:30 AM ET)")
        print(f"   Orders that gap beyond {LOO_SLIPPAGE_PCT*100:.0f}% will NOT fill")
    elif use_moo:
        print(f"\n📅 MOO orders will execute at next market open (9:30 AM ET)")
    
    print("\n" + "=" * 70)
    print("To execute these orders, run with --execute flag")
    if not use_moo and not use_loo:
        print("Add --loo flag for Limit-on-Open orders (recommended)")
        print("Add --moo flag for Market-on-Open orders")
    print("=" * 70 + "\n")


def execute_orders(api, orders: List[Dict], dry_run: bool = False, use_moo: bool = False, use_loo: bool = False) -> List[Dict]:
    """
    Submit orders to Alpaca.
    
    Args:
        api: Alpaca API connection
        orders: List of order dicts
        dry_run: If True, don't actually submit
        use_moo: If True, use Market-on-Open (time_in_force="opg")
        use_loo: If True, use Limit-on-Open (type="limit", time_in_force="opg")
    
    Returns list of order results.
    """
    results = []
    
    if use_loo:
        order_type_label = "LOO"
        order_type = "limit"
        time_in_force = "opg"
    elif use_moo:
        order_type_label = "MOO"
        order_type = "market"
        time_in_force = "opg"
    else:
        order_type_label = "MARKET"
        order_type = "market"
        time_in_force = "day"
    
    print("\n" + "=" * 70)
    print(f"EXECUTING ORDERS ({order_type_label})")
    print("=" * 70 + "\n")
    
    if use_loo:
        print(f"📅 Limit-on-Open orders with {LOO_SLIPPAGE_PCT*100:.0f}% buffer")
        print(f"   Will execute at next market open (9:30 AM ET)\n")
    elif use_moo:
        print("📅 Orders will be queued for next market open (9:30 AM ET)\n")
    
    for i, order in enumerate(orders, 1):
        symbol = order["symbol"]
        side = order["side"]
        qty = order["qty"]
        ref_price = order.get("ref_price", 0)
        
        if qty <= 0:
            continue
        
        # Calculate limit price for LOO
        if use_loo:
            limit_price = calculate_limit_price(ref_price, side)
            limit_price_str = format_limit_price(limit_price)
            print(f"[{i}/{len(orders)}] {side.upper()} {qty} {symbol} @ limit ${limit_price_str}...", end=" ")
        else:
            print(f"[{i}/{len(orders)}] {side.upper()} {qty} {symbol}...", end=" ")
        
        if dry_run:
            print("(dry run - skipped)")
            results.append({"symbol": symbol, "status": "dry_run"})
            continue
        
        try:
            # Build order parameters
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": order_type,
                "time_in_force": time_in_force,
            }
            
            # Add limit price for LOO orders
            if use_loo:
                order_params["limit_price"] = limit_price_str
            
            # Submit order
            submitted = api.submit_order(**order_params)
            print(f"✓ Order ID: {submitted.id}")
            
            result = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_id": submitted.id,
                "status": "submitted",
                "order_type": order_type_label,
                "time_in_force": time_in_force,
            }
            if use_loo:
                result["limit_price"] = float(limit_price_str)
                result["ref_price"] = ref_price
            
            results.append(result)
            
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
    print(f"Order Type: {order_type_label}")
    print(f"Submitted: {submitted}")
    print(f"Failed: {failed}")
    
    if use_loo and submitted > 0:
        print(f"\n📅 {submitted} LOO orders queued for next market open")
        print(f"   Non-fills (gaps > {LOO_SLIPPAGE_PCT*100:.0f}%) will expire - check status after open")
    elif use_moo and submitted > 0:
        print(f"\n📅 {submitted} orders queued for next market open")
    
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
    parser.add_argument("--moo", action="store_true", help="Use Market-on-Open orders (execute at next market open)")
    parser.add_argument("--loo", action="store_true", help="Use Limit-on-Open orders with 1%% buffer (recommended)")
    
    args = parser.parse_args()
    
    if not args.preview and not args.execute:
        print("Error: Must specify --preview or --execute")
        sys.exit(1)
    
    if args.moo and args.loo:
        print("Error: Cannot use both --moo and --loo. Choose one.")
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
        preview_orders(orders, account_info, use_moo=args.moo, use_loo=args.loo)
    elif args.execute:
        # Confirm before executing
        if args.loo:
            order_type = f"LOO (Limit-on-Open, {LOO_SLIPPAGE_PCT*100:.0f}% buffer)"
        elif args.moo:
            order_type = "MOO (Market-on-Open)"
        else:
            order_type = "MARKET (immediate)"
        
        print(f"\n⚠️  About to submit {len(orders)} {order_type} orders to Alpaca")
        print(f"    Account: {ALPACA_BASE_URL}")
        if args.moo or args.loo:
            print(f"    Execution: Next market open (9:30 AM ET)")
        if args.loo:
            print(f"    Non-fills: Orders gapping > {LOO_SLIPPAGE_PCT*100:.0f}% will expire")
        confirm = input("    Type 'YES' to confirm: ")
        
        if confirm != "YES":
            print("Cancelled.")
            sys.exit(0)
        
        results = execute_orders(api, orders, dry_run=args.dry_run, use_moo=args.moo, use_loo=args.loo)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"alpaca_orders_{timestamp}.csv"
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"\nOrder results saved to: {results_file}")


if __name__ == "__main__":
    main()