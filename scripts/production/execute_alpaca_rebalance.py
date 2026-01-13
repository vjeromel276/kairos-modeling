#!/usr/bin/env python3
"""
execute_alpaca_rebalance.py
===========================
Execute Kairos rebalance picks via Alpaca API.

Modes:
  1. Initial portfolio (no current positions) - buys all picks
  2. Rebalance (existing positions) - calculates delta from Alpaca positions

Order Types:
  - Paper trading: Defaults to intraday market orders (MOO simulation is unreliable)
  - Live trading: Defaults to Market-on-Open orders (submit Friday night, fills Monday 9:30 AM)
  - --intraday: Force immediate market orders
  - --force-moo: Force MOO orders even on paper trading (for testing)
  - --max-gap X: Skip stocks that gapped more than X% from Friday close (disaster protection)

Note on Paper Trading:
  Alpaca's paper trading environment does not properly simulate MOO orders.
  Approximately 27% of MOO orders expire without filling due to simulation bugs.
  Therefore, this script automatically uses intraday market orders for paper trading.
  See: https://forum.alpaca.markets/t/accurate-opg-and-cls-prices-for-paper-trading/3762

Workflow:
  Friday after close:
    1. Run pipeline, generate picks.csv
    2. Submit orders: python execute_alpaca_rebalance.py --picks picks.csv --execute
    3. Paper trading: executes immediately as market orders
       Live trading: queues MOO orders for Monday 9:30 AM open auction

Usage:
    # Preview what would happen (no orders submitted)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --preview

    # Execute (auto-detects paper vs live trading)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute

    # Execute with disaster protection (skip stocks gapping >5%)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute --max-gap 5

    # Force MOO orders on paper trading (for testing MOO behavior)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --execute --force-moo

    # Execute with custom portfolio value (default uses account equity)
    python scripts/production/execute_alpaca_rebalance.py --picks outputs/rebalance/2025-01-03/picks.csv --portfolio-value 100000 --execute

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

# Default max gap for disaster protection (None = no limit)
DEFAULT_MAX_GAP_PCT = None


def is_paper_trading() -> bool:
    """Check if we're using paper trading based on the API URL."""
    return "paper" in ALPACA_BASE_URL.lower()


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


def get_current_prices(api, symbols: List[str]) -> Dict[str, float]:
    """
    Fetch current/latest prices for symbols to check gap from reference price.
    Returns dict of symbol -> current price.
    """
    if not symbols:
        return {}
    
    prices = {}
    try:
        # Get latest trades for all symbols
        trades = api.get_latest_trades(symbols)
        for symbol, trade in trades.items():
            prices[symbol] = float(trade.price)
    except Exception as e:
        print(f"Warning: Could not fetch current prices: {e}")
    
    return prices


def check_gap_exceeded(ref_price: float, current_price: float, max_gap_pct: float, side: str) -> Tuple[bool, float]:
    """
    Check if current price has gapped beyond max_gap_pct from reference price.
    
    For buys: reject if current_price > ref_price * (1 + max_gap_pct)
    For sells: reject if current_price < ref_price * (1 - max_gap_pct)
    
    Returns (exceeded: bool, gap_pct: float)
    """
    if ref_price <= 0:
        return False, 0.0
    
    gap_pct = (current_price - ref_price) / ref_price * 100
    
    if side == "buy":
        exceeded = gap_pct > max_gap_pct
    else:  # sell
        exceeded = gap_pct < -max_gap_pct
    
    return exceeded, gap_pct


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


def preview_orders(orders: List[Dict], account_info: Dict, intraday: bool = False, max_gap_pct: float = None):
    """Print preview of orders without executing."""
    print("\n" + "=" * 70)
    print("ORDER PREVIEW (no orders will be submitted)")
    print("=" * 70)
    
    if intraday:
        order_type = "Market (immediate)"
    else:
        order_type = "Market-on-Open (MOO)"
    
    print(f"\nOrder Type: {order_type}")
    if max_gap_pct is not None:
        print(f"Max Gap Protection: {max_gap_pct}%")
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
    
    if not intraday:
        print(f"\n📅 MOO orders will execute at next market open (9:30 AM ET)")
    
    if max_gap_pct is not None:
        print(f"🛡️  Disaster protection: Orders skipped if gap > {max_gap_pct}% from reference price")
    
    print("\n" + "=" * 70)
    print("To execute these orders, run with --execute flag")
    if not intraday:
        print("Add --intraday flag for immediate market orders")
    print("=" * 70 + "\n")


def execute_orders(api, orders: List[Dict], dry_run: bool = False, intraday: bool = False, max_gap_pct: float = None) -> List[Dict]:
    """
    Submit orders to Alpaca.
    
    Args:
        api: Alpaca API connection
        orders: List of order dicts
        dry_run: If True, don't actually submit
        intraday: If True, use immediate market orders; else MOO
        max_gap_pct: If set, skip orders where current price gapped beyond this % from ref_price
    
    Returns list of order results.
    """
    results = []
    skipped_gap = 0
    
    if intraday:
        order_type_label = "MARKET"
        order_type = "market"
        time_in_force = "day"
    else:
        order_type_label = "MOO"
        order_type = "market"
        time_in_force = "opg"
    
    print("\n" + "=" * 70)
    print(f"EXECUTING ORDERS ({order_type_label})")
    print("=" * 70 + "\n")
    
    if not intraday:
        print("📅 Orders will be queued for next market open (9:30 AM ET)\n")
    
    # If max_gap protection enabled and intraday, fetch current prices
    current_prices = {}
    if max_gap_pct is not None and intraday:
        symbols = [o["symbol"] for o in orders]
        print(f"Fetching current prices for gap check (max {max_gap_pct}%)...")
        current_prices = get_current_prices(api, symbols)
        print(f"Got prices for {len(current_prices)} symbols\n")
    
    for i, order in enumerate(orders, 1):
        symbol = order["symbol"]
        side = order["side"]
        qty = order["qty"]
        ref_price = order.get("ref_price", 0)
        
        if qty <= 0:
            continue
        
        # Check gap protection (only for intraday orders with current prices)
        if max_gap_pct is not None and symbol in current_prices:
            exceeded, gap_pct = check_gap_exceeded(ref_price, current_prices[symbol], max_gap_pct, side)
            if exceeded:
                print(f"[{i}/{len(orders)}] {side.upper()} {qty} {symbol}... ⏭️ SKIPPED (gap {gap_pct:+.1f}% > {max_gap_pct}%)")
                results.append({
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "status": "skipped_gap",
                    "gap_pct": gap_pct,
                    "ref_price": ref_price,
                    "current_price": current_prices[symbol],
                })
                skipped_gap += 1
                continue
        
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
                "ref_price": ref_price,
            }
            
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
    if skipped_gap > 0:
        print(f"Skipped (gap exceeded): {skipped_gap}")
    print(f"Failed: {failed}")
    
    if not intraday and submitted > 0:
        print(f"\n📅 {submitted} MOO orders queued for next market open")
    
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
    parser.add_argument("--intraday", action="store_true", help="Force immediate market orders")
    parser.add_argument("--force-moo", action="store_true", help="Force MOO orders even on paper trading (for testing)")
    parser.add_argument("--max-gap", type=float, metavar="PCT", help="Skip orders if price gapped more than PCT%% from reference (disaster protection)")

    args = parser.parse_args()

    # Determine order mode: paper trading defaults to intraday due to MOO simulation bugs
    use_intraday = args.intraday
    if is_paper_trading() and not args.force_moo:
        use_intraday = True
        if not args.intraday:
            print("Note: Paper trading detected - using intraday orders (MOO simulation unreliable)")
            print("      Use --force-moo to override and test MOO behavior\n")
    
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
        preview_orders(orders, account_info, intraday=use_intraday, max_gap_pct=args.max_gap)
    elif args.execute:
        # Confirm before executing
        if use_intraday:
            order_type = "MARKET (immediate)"
        else:
            order_type = "MOO (Market-on-Open)"

        print(f"\n⚠️  About to submit {len(orders)} {order_type} orders to Alpaca")
        print(f"    Account: {ALPACA_BASE_URL}")
        if is_paper_trading():
            print(f"    Environment: PAPER TRADING")
        if not use_intraday:
            print(f"    Execution: Next market open (9:30 AM ET)")
        if args.max_gap:
            print(f"    Max gap protection: {args.max_gap}%")
        confirm = input("    Type 'YES' to confirm: ")

        if confirm != "YES":
            print("Cancelled.")
            sys.exit(0)

        results = execute_orders(api, orders, dry_run=args.dry_run, intraday=use_intraday, max_gap_pct=args.max_gap)
        
        # Save results in same directory as picks.csv
        picks_dir = os.path.dirname(args.picks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"alpaca_orders_{timestamp}.csv"
        results_file = os.path.join(picks_dir, results_filename) if picks_dir else results_filename
        pd.DataFrame(results).to_csv(results_file, index=False)
        print(f"\nOrder results saved to: {results_file}")


if __name__ == "__main__":
    main()