#!/usr/bin/env python3
"""
measure_live_slippage.py
========================
Pulls actual Alpaca fill data for every order submitted during the live
rebalance history, joins to the reference prices captured at order
submission (outputs/rebalance/*/alpaca_orders_*.csv) and computes the
realized per-trade slippage.

Primary question this answers: **is our 15 bps assumed cost realistic?**

If mean round-trip slippage is <12 bps → CPPI stays optimal.
If mean round-trip slippage is >12 bps → dd_linear would have won.

Read-only: queries Alpaca for order details, writes a summary CSV + JSON.

Usage:
  python scripts/evaluation/measure_live_slippage.py \\
      --rebalance-dir outputs/rebalance \\
      --output-dir outputs/evaluation/live_slippage
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("live_slippage")

# Credentials fall back to the same paper-trading defaults used elsewhere.
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY",
                              "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


def collect_orders(rebalance_dir: Path) -> pd.DataFrame:
    """Load every alpaca_orders_*.csv in every rebalance folder, tag with date."""
    rows = []
    for csv in sorted(rebalance_dir.glob("*/alpaca_orders_*.csv")):
        rebal_date = csv.parent.name  # folder name = YYYY-MM-DD
        df = pd.read_csv(csv)
        df["rebalance_date"] = rebal_date
        df["source_file"] = csv.name
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    log.info("Collected %d order rows across %d rebalance files",
             len(out), len(rows))
    return out


def fetch_all_fills(api, after: str = "2026-01-01", page_size: int = 100) -> pd.DataFrame:
    """
    Pull every FILL activity via the activities API, paginated.
    Much faster than individual order lookups (one batched request per 100 fills).
    """
    log.info("Fetching FILL activities since %s (paginated)...", after)
    all_fills = []
    page_token = None
    t0 = time.time()
    while True:
        kwargs = {"activity_types": "FILL", "after": after, "page_size": page_size,
                  "direction": "asc"}
        if page_token:
            kwargs["page_token"] = page_token
        acts = api.get_activities(**kwargs)
        if not acts:
            break
        for a in acts:
            all_fills.append({
                "order_id": a.order_id,
                "symbol": a.symbol,
                "side_api": a.side,
                "filled_qty": float(a.qty or 0),
                "filled_avg_price": float(a.price or 0) if a.price else np.nan,
                "filled_at": str(a.transaction_time),
                "activity_id": a.id,
            })
        page_token = acts[-1].id  # continue from the last activity
        if len(acts) < page_size:
            break
    log.info("  got %d fills in %.1fs", len(all_fills), time.time() - t0)
    return pd.DataFrame(all_fills)


def compute_slippage(orders: pd.DataFrame) -> pd.DataFrame:
    """
    For each filled order:
      slippage_bps = (fill_price - ref_price) / ref_price × 10000 × side_sign
        side_sign = +1 for BUY (positive slippage = paid more than expected)
        side_sign = -1 for SELL (positive slippage = got less than expected)

    Round-trip slippage = sum of buy and sell slippage on the same notional.
    We'll report per-leg (since buy and sell use different ref prices).
    """
    orders = orders.copy()
    orders["ref_price"] = pd.to_numeric(orders.get("ref_price"), errors="coerce")
    orders["filled_avg_price"] = pd.to_numeric(orders["filled_avg_price"], errors="coerce")
    orders["filled_qty"] = pd.to_numeric(orders["filled_qty"], errors="coerce")

    filled = orders[(orders["filled_qty"] > 0) & (orders["filled_avg_price"] > 0)
                    & orders["ref_price"].notna()].copy()

    sign = np.where(filled["side"].str.lower() == "buy", 1.0, -1.0)
    slip = (filled["filled_avg_price"] - filled["ref_price"]) / filled["ref_price"]
    filled["slippage_bps"] = slip * 10000 * sign
    filled["fill_notional"] = filled["filled_qty"] * filled["filled_avg_price"]
    return filled


def summarize(filled: pd.DataFrame) -> dict:
    if filled.empty:
        return {"n_fills": 0}
    wtd_mean_bps = float(
        (filled["slippage_bps"] * filled["fill_notional"]).sum()
        / filled["fill_notional"].sum()
    )
    return {
        "n_fills": int(len(filled)),
        "total_notional": round(float(filled["fill_notional"].sum()), 2),
        "mean_slippage_bps": round(float(filled["slippage_bps"].mean()), 2),
        "median_slippage_bps": round(float(filled["slippage_bps"].median()), 2),
        "notional_weighted_slippage_bps": round(wtd_mean_bps, 2),
        "std_slippage_bps": round(float(filled["slippage_bps"].std()), 2),
        "p25_slippage_bps": round(float(filled["slippage_bps"].quantile(0.25)), 2),
        "p75_slippage_bps": round(float(filled["slippage_bps"].quantile(0.75)), 2),
        "buy_mean_bps": round(float(
            filled[filled["side"].str.lower() == "buy"]["slippage_bps"].mean()
        ), 2),
        "sell_mean_bps": round(float(
            filled[filled["side"].str.lower() == "sell"]["slippage_bps"].mean()
        ), 2),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Measure realized Alpaca slippage")
    p.add_argument("--rebalance-dir", type=Path,
                   default=Path("outputs/rebalance"))
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/evaluation/live_slippage"))
    p.add_argument("--max-orders", type=int, default=0,
                   help="Cap lookups for testing (0 = no cap)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    orders = collect_orders(args.rebalance_dir)
    if orders.empty:
        log.error("No order files found under %s", args.rebalance_dir)
        return 1

    # Only look up orders that were submitted (have order_id) — failed orders
    # never had a fill.
    submittable = orders[orders["order_id"].notna() & (orders["order_id"] != "")].copy()
    log.info("  %d submitted orders (of %d total) to look up", len(submittable), len(orders))

    if args.max_orders > 0:
        submittable = submittable.head(args.max_orders)
        log.info("  (limiting to %d for testing)", len(submittable))

    api = get_api()

    # Activities-API approach: one batched pull, then join by order_id.
    # An order may have multiple fills (partial fills) — aggregate first.
    fills = fetch_all_fills(api, after="2026-01-01")
    if fills.empty:
        log.warning("No FILL activities returned. Check API keys / date range.")
        return 1

    # If an order had multiple partial fills, average their prices weighted by qty
    fills_agg = (fills.groupby("order_id")
                 .apply(lambda g: pd.Series({
                     "filled_qty": g["filled_qty"].sum(),
                     "filled_avg_price": (g["filled_qty"] * g["filled_avg_price"]).sum()
                                         / g["filled_qty"].sum(),
                     "filled_at": g["filled_at"].min(),
                     "n_fills_per_order": len(g),
                 }))
                 .reset_index())
    log.info("  %d unique filled order_ids (from %d fill events)",
             len(fills_agg), len(fills))

    joined = submittable.merge(fills_agg, on="order_id", how="left")
    joined["api_status"] = np.where(joined["filled_qty"].fillna(0) > 0, "filled",
                                    "not_filled")

    filled = compute_slippage(joined)
    log.info("  %d filled orders (out of %d looked up)",
             len(filled), len(submittable))

    # Write per-trade detail
    out_csv = args.output_dir / "live_slippage_trades.csv"
    filled.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)

    # Aggregate stats
    overall = summarize(filled)
    by_side = {
        "buys": summarize(filled[filled["side"].str.lower() == "buy"]),
        "sells": summarize(filled[filled["side"].str.lower() == "sell"]),
    }
    by_date = {}
    for d in sorted(filled["rebalance_date"].unique()):
        by_date[d] = summarize(filled[filled["rebalance_date"] == d])

    # Top-line: what does this mean for the cost assumption?
    wtd = overall.get("notional_weighted_slippage_bps", float("nan"))
    one_way = (by_side["buys"].get("notional_weighted_slippage_bps", 0)
               + by_side["sells"].get("notional_weighted_slippage_bps", 0))
    round_trip_est = by_side["buys"].get("notional_weighted_slippage_bps", 0) \
                   + by_side["sells"].get("notional_weighted_slippage_bps", 0)

    summary = {
        "overall": overall,
        "by_side": by_side,
        "by_rebalance_date": by_date,
        "cost_assumption_check": {
            "per_trade_bps_notional_weighted": wtd,
            "round_trip_estimate_bps": round_trip_est,
            "backtest_assumption_bps": 15,
            "break_even_for_cppi_bps": 12,
            "verdict": (
                "CPPI stays optimal" if round_trip_est < 12
                else "dd_linear would win" if round_trip_est > 12
                else "on the boundary"
            ),
        },
    }

    with open(args.output_dir / "live_slippage_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Print the headline
    def fmt(x, default="--"):
        return f"{x:+.2f}" if isinstance(x, (int, float)) and not np.isnan(x) else default

    print("\n" + "=" * 72)
    print("REALIZED SLIPPAGE — ALPACA LIVE TRADES (2026-01-02 to present)")
    print("=" * 72)
    print(f"  n filled orders:        {overall.get('n_fills', 0):,}")
    print(f"  total fill notional:    ${overall.get('total_notional', 0):,.0f}")
    print(f"  per-trade mean (bps):   {fmt(overall.get('mean_slippage_bps'))}")
    print(f"  per-trade median (bps): {fmt(overall.get('median_slippage_bps'))}")
    print(f"  per-trade wtd mean:     {fmt(overall.get('notional_weighted_slippage_bps'))} bps")
    print(f"  buy mean:               {fmt(by_side['buys'].get('notional_weighted_slippage_bps'))} bps")
    print(f"  sell mean:              {fmt(by_side['sells'].get('notional_weighted_slippage_bps'))} bps")
    print(f"  ESTIMATED ROUND-TRIP:   {fmt(round_trip_est)} bps")
    print()
    print(f"  Backtest assumed:       15 bps round-trip")
    print(f"  CPPI break-even:        12 bps (below this, CPPI wins)")
    print(f"  Verdict:                {summary['cost_assumption_check']['verdict']}")
    print("=" * 72)
    print(f"\nDetail: {out_csv}")
    print(f"Summary: {args.output_dir / 'live_slippage_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
