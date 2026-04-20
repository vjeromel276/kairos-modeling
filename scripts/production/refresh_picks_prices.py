#!/usr/bin/env python3
"""
refresh_picks_prices.py
=======================
Overwrites the `price` / `target_value` / `shares` columns in a picks.csv
with live Alpaca quotes. Intended to run right before executing the
rebalance (e.g. Monday 9:25am pre-open), so slippage and position sizing
measure against fill-time prices rather than stale Friday-close prices.

Nothing about signal generation or the pipeline changes — this script
touches only the execution-side price inputs.

Usage:
  python scripts/production/refresh_picks_prices.py \\
      --picks outputs/rebalance/2026-04-17/picks.csv

Creates a backup at picks.csv.bak before overwriting.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("refresh_prices")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY", "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB"
)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


def fetch_quote(api, ticker: str) -> dict:
    """Return {bid, ask, mid, last, staleness_s} for a single ticker."""
    try:
        q = api.get_latest_quote(ticker)
        bid = float(q.bid_price) if q.bid_price else None
        ask = float(q.ask_price) if q.ask_price else None
        mid = None
        if bid and ask and bid > 0 and ask > 0:
            mid = (bid + ask) / 2.0
        out = {"bid": bid, "ask": ask, "mid": mid}
    except Exception as e:
        log.warning("  quote fail %s: %s", ticker, str(e)[:80])
        out = {"bid": None, "ask": None, "mid": None}

    # Fallback to latest trade when the NBBO is missing (common pre-open)
    if out["mid"] is None:
        try:
            t = api.get_latest_trade(ticker)
            out["last"] = float(t.price) if t.price else None
            if out["last"]:
                out["mid"] = out["last"]
        except Exception:
            out["last"] = None
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description="Refresh picks.csv prices with live Alpaca quotes"
    )
    p.add_argument("--picks", required=True, type=Path,
                   help="Path to picks.csv to update in place")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the refreshed prices but don't overwrite the file")
    p.add_argument("--no-backup", action="store_true",
                   help="Skip writing picks.csv.bak")
    args = p.parse_args()

    if not args.picks.exists():
        log.error("Picks file not found: %s", args.picks)
        return 1

    df = pd.read_csv(args.picks)
    if "ticker" not in df.columns:
        log.error("picks.csv missing 'ticker' column")
        return 1
    log.info("Loaded %d picks from %s", len(df), args.picks)

    api = get_api()

    t0 = time.time()
    new_prices, kinds = [], []
    for i, row in df.iterrows():
        q = fetch_quote(api, row["ticker"])
        # Prefer mid, else last, else keep the stale price
        if q["mid"] is not None:
            new_prices.append(q["mid"])
            kinds.append("mid" if q.get("bid") else "last")
        else:
            new_prices.append(row["price"])
            kinds.append("stale")
        if (i + 1) % 25 == 0 or (i + 1) == len(df):
            log.info("  %d/%d quotes fetched (%.1fs)", i + 1, len(df), time.time() - t0)

    df["price_refreshed"] = new_prices
    df["price_source"] = kinds

    # Diagnostic — how much did each price move
    old_prices = df["price"].astype(float)
    df["price_drift_bps"] = ((df["price_refreshed"] - old_prices)
                             / old_prices * 10000).round(1)

    # Shares recompute: target_value = weight × portfolio_value is fixed
    # so when price changes, shares change. weight is the invariant.
    # We need portfolio_value. If target_value is present, use it;
    # otherwise infer from the existing shares × old price.
    if "target_value" in df.columns:
        implied_pv = (df["target_value"] / df["weight"]).replace(
            [float("inf"), float("nan")], None
        ).median()
        log.info("Implied portfolio_value (from target_value / weight median): $%.0f",
                 implied_pv if implied_pv else 0)
        df["target_value_new"] = (df["weight"] * implied_pv).round(2)
    else:
        df["target_value_new"] = df["target_value"] if "target_value" in df.columns \
            else (df["shares"] * old_prices)

    df["shares_new"] = (df["target_value_new"] / df["price_refreshed"]).fillna(0).astype(int)

    # Summary
    n_stale = (df["price_source"] == "stale").sum()
    drift = df["price_drift_bps"].dropna()
    log.info("\n=== Price drift summary (new_mid vs old Friday close) ===")
    log.info("  stale (no quote available): %d", n_stale)
    log.info("  mean drift:  %+.1f bps", drift.mean())
    log.info("  median:      %+.1f bps", drift.median())
    log.info("  |drift| p90: %.1f bps", drift.abs().quantile(0.9))
    log.info("  max drift:   %+.1f bps (%s)",
             drift.abs().max(),
             df.loc[drift.abs().idxmax(), "ticker"])

    # Largest share-count changes
    df["share_delta"] = df["shares_new"] - df["shares"]
    big_changes = df.nlargest(5, "share_delta", keep="all").head(5)
    log.info("\nLargest upward share changes (bigger weekend moves):")
    for _, r in big_changes.iterrows():
        log.info("  %-6s  old=%d → new=%d  (price %.2f → %.2f, drift %+.1f bps)",
                 r["ticker"], int(r["shares"]), int(r["shares_new"]),
                 r["price"], r["price_refreshed"], r["price_drift_bps"])

    if args.dry_run:
        log.info("--dry-run: not overwriting %s", args.picks)
        return 0

    if not args.no_backup:
        bak = args.picks.with_suffix(".csv.bak")
        shutil.copy(args.picks, bak)
        log.info("Backup: %s", bak)

    # Overwrite the production columns
    df["price"] = df["price_refreshed"].round(2)
    if "target_value" in df.columns:
        df["target_value"] = df["target_value_new"]
    df["shares"] = df["shares_new"].astype(int)

    drop_cols = [
        "price_refreshed", "price_source", "price_drift_bps",
        "target_value_new", "shares_new", "share_delta",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    df.to_csv(args.picks, index=False)
    log.info("Overwrote %s with refreshed prices (%d rows)", args.picks, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
