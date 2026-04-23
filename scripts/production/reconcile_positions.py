#!/usr/bin/env python3
"""
reconcile_positions.py
======================
Weekly diff between (picks.csv target, prior picks.csv, current Alpaca
positions). Identifies:

  - ZOMBIE positions: held on Alpaca, Alpaca won't let us sell them
    (e.g. delisted stocks like SEE/SMLR/OS — "asset is not active").
  - ORPHAN positions: held on Alpaca but not in current picks or prior
    picks — picked up somehow (mis-fill, old rebalance) and never cleaned.
  - MISSING positions: in picks.csv (supposed to hold) but not actually
    held on Alpaca (order never filled, partial fill).
  - EFFECTIVE EQUITY: raw Alpaca equity minus zombie value — this is what
    CPPI / rebalance sizing should actually use.

Writes a JSON report. Non-destructive — only reads and queries.

Usage:
  python scripts/production/reconcile_positions.py \\
      --picks outputs/rebalance/YYYY-MM-DD/picks.csv \\
      --output-dir outputs/evaluation/reconcile
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("reconcile")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "PK347Y7OMCULH3KC5MALII6ZWP")
ALPACA_SECRET_KEY = os.getenv(
    "ALPACA_SECRET_KEY", "7vceesTCANBZXXjMjEuGs1a8N1YjkAudj4aKUUXofRHB"
)
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")


def get_api():
    import alpaca_trade_api as tradeapi
    return tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL)


def check_ticker_tradable(api, ticker: str) -> bool:
    try:
        a = api.get_asset(ticker)
        return bool(getattr(a, "tradable", False))
    except Exception:
        return False


def fetch_state(api):
    """Return (account_dict, positions_df)."""
    a = api.get_account()
    account = {
        "equity": float(a.equity),
        "last_equity": float(a.last_equity),
        "cash": float(a.cash),
        "buying_power": float(a.buying_power),
        "long_market_value": float(a.long_market_value),
    }
    pos = api.list_positions()
    rows = []
    for p in pos:
        rows.append({
            "ticker": p.symbol,
            "qty": int(p.qty),
            "market_value": float(p.market_value or 0),
            "avg_entry_price": float(p.avg_entry_price or 0),
            "current_price": float(p.current_price or 0),
            "unrealized_pl": float(p.unrealized_pl or 0),
        })
    return account, pd.DataFrame(rows)


def find_prior_picks(picks_path: Path) -> Path | None:
    """Find the most recent rebalance folder's picks.csv before picks_path's date."""
    parent = picks_path.parent.parent  # outputs/rebalance/
    if not parent.exists():
        return None
    current_date = picks_path.parent.name  # YYYY-MM-DD
    siblings = sorted(
        d for d in parent.iterdir()
        if d.is_dir() and d.name < current_date and (d / "picks.csv").exists()
    )
    return siblings[-1] / "picks.csv" if siblings else None


def main() -> int:
    p = argparse.ArgumentParser(
        description="Reconcile picks vs Alpaca positions"
    )
    p.add_argument("--picks", required=True, type=Path,
                   help="Current rebalance picks.csv")
    p.add_argument("--prior-picks", type=Path, default=None,
                   help="Optional — if omitted, auto-finds prior rebalance folder")
    p.add_argument("--output-dir", type=Path,
                   default=Path("outputs/evaluation/reconcile"))
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.picks.exists():
        log.error("picks not found: %s", args.picks)
        return 1

    picks = pd.read_csv(args.picks)
    picks_set = set(picks["ticker"].astype(str).str.upper())
    log.info("Loaded current picks: %d tickers from %s", len(picks_set), args.picks)

    prior_path = args.prior_picks or find_prior_picks(args.picks)
    prior_set = set()
    if prior_path and Path(prior_path).exists():
        prior_set = set(pd.read_csv(prior_path)["ticker"].astype(str).str.upper())
        log.info("Prior picks: %d tickers from %s", len(prior_set), prior_path)
    else:
        log.info("No prior picks found — zombie detection will be coarser")

    api = get_api()
    account, positions = fetch_state(api)
    log.info("Alpaca: %d positions, equity=$%.2f, long_mv=$%.2f",
             len(positions), account["equity"], account["long_market_value"])

    held_set = set(positions["ticker"].astype(str).str.upper())

    # Classify each held position
    log.info("Checking tradability for each held position...")
    positions["tradable"] = positions["ticker"].apply(
        lambda t: check_ticker_tradable(api, t)
    )
    positions["in_current_picks"] = positions["ticker"].isin(picks_set)
    positions["in_prior_picks"] = positions["ticker"].isin(prior_set) if prior_set else False

    # Zombies: held, not tradable
    zombies = positions[~positions["tradable"]].copy()
    # Orphans: held, tradable, not in current or prior picks
    orphans = positions[
        positions["tradable"]
        & ~positions["in_current_picks"]
        & ~positions["in_prior_picks"]
        & bool(prior_set)  # only meaningful if we have prior_set
    ].copy() if prior_set else pd.DataFrame()
    # Missing: in picks, not held
    missing = sorted(picks_set - held_set)

    zombie_value = float(zombies["market_value"].sum()) if len(zombies) else 0.0
    effective_equity = account["equity"] - zombie_value

    log.info("\n" + "=" * 68)
    log.info("RECONCILIATION")
    log.info("=" * 68)
    log.info(f"  Total held:          {len(positions)}")
    log.info(f"  In current picks:    {int(positions['in_current_picks'].sum())}")
    log.info(f"  In prior picks:      {int(positions['in_prior_picks'].sum())}")
    log.info(f"  ZOMBIES (untradable):  {len(zombies)}  (value ${zombie_value:,.2f})")
    log.info(f"  ORPHANS:               {len(orphans)}")
    log.info(f"  MISSING from Alpaca:   {len(missing)}  ({missing[:5]}{'...' if len(missing) > 5 else ''})")
    log.info(f"  Raw equity:          ${account['equity']:,.2f}")
    log.info(f"  Zombie value:        ${zombie_value:,.2f}  ({zombie_value/account['equity']*100:.2f}% of equity)")
    log.info(f"  EFFECTIVE EQUITY:    ${effective_equity:,.2f}")

    if len(zombies):
        log.info("\n  Zombies detail:")
        for _, z in zombies.iterrows():
            log.info(f"    {z['ticker']:<10}  qty={z['qty']:>5}  "
                     f"mv=${z['market_value']:>9,.2f}  avg=${z['avg_entry_price']:>7,.2f}")

    if len(orphans):
        log.info("\n  Orphans detail:")
        for _, o in orphans.iterrows():
            log.info(f"    {o['ticker']:<10}  qty={o['qty']:>5}  "
                     f"mv=${o['market_value']:>9,.2f}")

    # Write JSON report
    report = {
        "generated_at": datetime.now().isoformat(),
        "current_picks_path": str(args.picks),
        "prior_picks_path": str(prior_path) if prior_path else None,
        "account": account,
        "n_positions": int(len(positions)),
        "n_in_current_picks": int(positions["in_current_picks"].sum()),
        "n_in_prior_picks": int(positions["in_prior_picks"].sum()),
        "zombies": zombies[["ticker", "qty", "market_value",
                             "avg_entry_price"]].to_dict(orient="records"),
        "zombie_total_value": zombie_value,
        "orphans": orphans[["ticker", "qty", "market_value"]].to_dict(orient="records")
            if len(orphans) else [],
        "missing_from_alpaca": missing,
        "effective_equity": effective_equity,
        "effective_equity_pct_of_raw": effective_equity / account["equity"]
            if account["equity"] > 0 else None,
    }

    out_path = args.output_dir / f"reconcile_{args.picks.parent.name}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"\nWrote report: {out_path}")

    # Also a "latest" symlink for tooling / CPPI lookup
    latest = args.output_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    with open(latest, "w") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Also wrote: {latest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
