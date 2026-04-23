#!/usr/bin/env python3
"""
check_tradability.py
====================
Queries Alpaca for each ticker in picks.csv and drops/flags assets that
are not actually tradable (delisted, inactive, not fractionable when
fractions are required, etc). Prevents "asset is not active" failures
at execution time.

Typical failures seen in the wild: SEE, SMLR, OS (delisted), 436CVR021
(contingent value right — technically an asset but Alpaca rejects trades).

Usage:
  # Default: rewrite picks.csv in place (keeps a .bak), drop untradable rows
  python scripts/production/check_tradability.py \\
      --picks outputs/rebalance/YYYY-MM-DD/picks.csv

  # Dry-run — shows which tickers would be dropped without touching file
  python scripts/production/check_tradability.py --picks ... --dry-run

  # Flag only (adds a `tradable` column but keeps all rows)
  python scripts/production/check_tradability.py --picks ... --flag-only
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
log = logging.getLogger("check_tradability")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY", "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB"
)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


def check_ticker(api, ticker: str) -> dict:
    """Return a dict describing Alpaca's view of this ticker."""
    try:
        a = api.get_asset(ticker)
        return {
            "ticker": ticker,
            "tradable": bool(getattr(a, "tradable", False)),
            "status": getattr(a, "status", None),
            "exchange": getattr(a, "exchange", None),
            "asset_class": getattr(a, "class", None) or getattr(a, "asset_class", None),
            "reason": None,
        }
    except Exception as e:
        msg = str(e)[:120]
        return {"ticker": ticker, "tradable": False, "status": "not_found",
                "exchange": None, "asset_class": None, "reason": msg}


def main() -> int:
    p = argparse.ArgumentParser(
        description="Filter untradable tickers from picks.csv"
    )
    p.add_argument("--picks", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true",
                   help="Report without modifying picks.csv")
    p.add_argument("--flag-only", action="store_true",
                   help="Keep all rows; add tradable/status columns only")
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()

    if not args.picks.exists():
        log.error("picks file not found: %s", args.picks)
        return 1

    df = pd.read_csv(args.picks)
    tickers = df["ticker"].astype(str).str.upper().tolist()
    log.info("Checking tradability on %d tickers...", len(tickers))

    api = get_api()
    t0 = time.time()
    results = []
    for i, t in enumerate(tickers, 1):
        results.append(check_ticker(api, t))
        if i % 25 == 0 or i == len(tickers):
            log.info("  %d/%d checked (%.1fs)", i, len(tickers), time.time() - t0)

    rdf = pd.DataFrame(results)
    untradable = rdf[~rdf["tradable"]].copy()
    log.info("\n=== Tradability summary ===")
    log.info("  tradable:    %d", int(rdf["tradable"].sum()))
    log.info("  untradable:  %d", int((~rdf["tradable"]).sum()))
    if len(untradable):
        log.info("\n  Untradable tickers:")
        for _, r in untradable.iterrows():
            log.info("    %-10s  status=%-12s  reason=%s",
                     r["ticker"], r["status"] or "-",
                     (r["reason"] or "")[:80])

    # Merge tradable flag onto picks
    merged = df.merge(rdf[["ticker", "tradable", "status"]], on="ticker", how="left")

    if args.dry_run:
        log.info("--dry-run: not modifying %s", args.picks)
        return 0

    if not args.no_backup:
        bak = args.picks.with_suffix(".csv.tradability.bak")
        shutil.copy(args.picks, bak)
        log.info("Backup: %s", bak)

    if args.flag_only:
        merged.to_csv(args.picks, index=False)
        log.info("Flag-only mode — wrote %d rows (all kept, "
                 "tradable/status columns added)", len(merged))
        return 0

    kept = merged[merged["tradable"].astype(bool)].drop(
        columns=[c for c in ("tradable", "status") if c in merged.columns]
    )
    dropped = merged[~merged["tradable"].astype(bool)]
    kept.to_csv(args.picks, index=False)
    log.info("Dropped %d untradable tickers; wrote %d rows",
             len(dropped), len(kept))
    if len(dropped) > 0:
        out = args.picks.with_name("untradable_" + args.picks.name)
        dropped.to_csv(out, index=False)
        log.info("Saved dropped tickers to %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
