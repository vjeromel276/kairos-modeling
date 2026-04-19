#!/usr/bin/env python3
"""
cost_sensitivity_suite.py
=========================
Three verifications in one pass:

  1. Spread sensitivity sweep — run the full STRATEGIES dict at spread levels
     [0, 5, 10, 15, 20, 30] bps. Find the break-even cost where CPPI-prod
     ties baseline.
  2. Impact model variant — run at spread=15 with a linear impact coefficient.
     Likely widens the CPPI-vs-baseline loss.
  3. CPPI (floor, multiplier) grid under costs — re-run the gross-winner
     grid at spread=15 to see if a lower-turnover CPPI config survives.

Produces a single summary CSV with all results tagged by (cost regime,
strategy) so the next decision is evidence-based.

Usage:
  python scripts/evaluation/cost_sensitivity_suite.py --db data/kairos.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluation.backtest_capital_allocation import (  # noqa: E402
    STRATEGIES, annualize, calmar, max_drawdown, run_backtest,
)
from scripts.evaluation.cppi_grid_sweep import build_grid  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("cost_suite")

SPREADS = [0, 5, 10, 15, 20, 30]


def eval_strategy(bt: pd.DataFrame, name: str, cfg: dict, spread: float,
                  impact: float) -> dict:
    ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], 5)
    eq = (1 + bt["port_ret"]).cumprod()
    mdd = max_drawdown(eq)
    cal = calmar(ann_ret, mdd)
    ann_turnover = bt["turnover"].sum() / max(1, len(bt)) * (252 / 5) \
        if "turnover" in bt.columns else 0.0
    ann_cost = bt["cost_scaled"].sum() / max(1, len(bt)) * (252 / 5) \
        if "cost_scaled" in bt.columns else 0.0
    return {
        "strategy": name,
        "fn": cfg["fn"],
        "floor": cfg["params"].get("floor_pct"),
        "mult": cfg["params"].get("multiplier"),
        "spread_bps": spread,
        "impact_coef": impact,
        "sharpe": round(sharpe, 4),
        "ann_return": round(ann_ret, 4),
        "max_drawdown": round(mdd, 4),
        "calmar": round(cal, 3),
        "avg_allocation": round(float(bt["allocation"].mean()), 3),
        "annual_turnover": round(ann_turnover, 2),
        "annual_cost": round(ann_cost, 4),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--alpha-column", default="alpha_ml_v2_tuned_clf")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2025-11-01")
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--output-dir", default="outputs/evaluation/cost_suite")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading panel + regime...")
    con = duckdb.connect(args.db, read_only=True)
    df = con.execute(f"""
        SELECT fm.ticker, fm.date,
               fm.{args.alpha_column} AS alpha,
               fm.ret_5d_f AS target,
               fm.vol_blend, fm.adv_20,
               t.sector
        FROM feat_matrix_v2 fm
        LEFT JOIN (SELECT DISTINCT ticker, sector FROM tickers
                   WHERE sector IS NOT NULL AND ticker != 'N/A') t USING (ticker)
        WHERE fm.{args.alpha_column} IS NOT NULL
          AND fm.ret_5d_f IS NOT NULL
          AND fm.vol_blend IS NOT NULL
          AND fm.adv_20 IS NOT NULL
          AND fm.date BETWEEN DATE '{args.start_date}' AND DATE '{args.end_date}'
        ORDER BY fm.date, fm.ticker
    """).fetchdf()
    regime_df = con.execute("SELECT date, vol_regime FROM regime_history_academic").fetchdf()
    con.close()
    log.info("  %d rows, %d regime rows", len(df), len(regime_df))

    all_rows = []

    # ------------------------------------------------------------------------
    # (1) Spread sensitivity sweep — all STRATEGIES × spread levels
    # ------------------------------------------------------------------------
    log.info("\n=== (1) SPREAD SENSITIVITY SWEEP ===")
    for spread in SPREADS:
        log.info("\n-- spread_bps = %g --", spread)
        for name, cfg in STRATEGIES.items():
            bt = run_backtest(df.copy(), regime_df,
                              strategy_name=name, strategy_config=cfg,
                              target_vol=args.target_vol,
                              spread_bps=spread, impact_coef=0.0)
            if bt.empty:
                continue
            row = eval_strategy(bt, name, cfg, spread, 0.0)
            row["regime"] = "spread_sweep"
            all_rows.append(row)
            log.info("  %-22s  Sharpe=%+.3f  CAGR=%+.1f%%  DD=%+.1f%%  Calmar=%+.2f  Cost=%.2f%%",
                     name, row["sharpe"], row["ann_return"] * 100,
                     row["max_drawdown"] * 100, row["calmar"],
                     row["annual_cost"] * 100)

    # ------------------------------------------------------------------------
    # (2) Impact model variant — spread=15 + linear impact
    # ------------------------------------------------------------------------
    log.info("\n=== (2) IMPACT MODEL (spread=15, impact_coef=0.1, book=$100k) ===")
    for name, cfg in STRATEGIES.items():
        bt = run_backtest(df.copy(), regime_df,
                          strategy_name=name, strategy_config=cfg,
                          target_vol=args.target_vol,
                          spread_bps=15.0, impact_coef=0.1,
                          portfolio_value=100_000)
        if bt.empty:
            continue
        row = eval_strategy(bt, name, cfg, 15.0, 0.1)
        row["regime"] = "impact_100k"
        all_rows.append(row)
        log.info("  %-22s  Sharpe=%+.3f  CAGR=%+.1f%%  Calmar=%+.2f  Cost=%.2f%%",
                 name, row["sharpe"], row["ann_return"] * 100,
                 row["calmar"], row["annual_cost"] * 100)

    # Also at $1M and $10M — impact scales with size
    for pv in (1_000_000, 10_000_000):
        log.info("\n-- impact at $%s book --", f"{int(pv):,}")
        for name, cfg in STRATEGIES.items():
            bt = run_backtest(df.copy(), regime_df,
                              strategy_name=name, strategy_config=cfg,
                              target_vol=args.target_vol,
                              spread_bps=15.0, impact_coef=0.1,
                              portfolio_value=pv)
            if bt.empty:
                continue
            row = eval_strategy(bt, name, cfg, 15.0, 0.1)
            row["regime"] = f"impact_{int(pv)}"
            row["portfolio_value"] = pv
            all_rows.append(row)
            log.info("  %-22s  Sharpe=%+.3f  CAGR=%+.1f%%  Calmar=%+.2f  Cost=%.2f%%",
                     name, row["sharpe"], row["ann_return"] * 100,
                     row["calmar"], row["annual_cost"] * 100)

    # ------------------------------------------------------------------------
    # (3) CPPI grid under costs — same peak-lock grid at spread=15
    # ------------------------------------------------------------------------
    log.info("\n=== (3) CPPI GRID UNDER 15 bps ===")
    grid = build_grid("peak_lock")
    for name, cfg in grid.items():
        bt = run_backtest(df.copy(), regime_df,
                          strategy_name=name, strategy_config=cfg,
                          target_vol=args.target_vol,
                          spread_bps=15.0, impact_coef=0.0)
        if bt.empty:
            continue
        row = eval_strategy(bt, name, cfg, 15.0, 0.0)
        row["regime"] = "cppi_grid_net15"
        all_rows.append(row)
        log.info("  %-32s  Sharpe=%+.3f  CAGR=%+.1f%%  Calmar=%+.2f  Cost=%.2f%%",
                 name, row["sharpe"], row["ann_return"] * 100,
                 row["calmar"], row["annual_cost"] * 100)

    # ------------------------------------------------------------------------
    # Aggregate + write
    # ------------------------------------------------------------------------
    rdf = pd.DataFrame(all_rows)
    out_csv = out / "cost_sensitivity_all_results.csv"
    rdf.to_csv(out_csv, index=False)
    log.info("\nWrote %s (%d rows)", out_csv, len(rdf))

    # Key view: spread sensitivity — strategies × spreads matrix
    log.info("\n" + "=" * 90)
    log.info("SPREAD SENSITIVITY: net Calmar by strategy × spread (bps)")
    log.info("=" * 90)
    focus = rdf[rdf["regime"] == "spread_sweep"].pivot(
        index="strategy", columns="spread_bps", values="calmar"
    )
    focus = focus.reindex(columns=SPREADS)
    focus = focus.sort_values(15.0, ascending=False)
    log.info("\n%s", focus.round(3).to_string())

    log.info("\n" + "=" * 90)
    log.info("SPREAD SENSITIVITY: net CAGR by strategy × spread (bps)")
    log.info("=" * 90)
    focus_ret = rdf[rdf["regime"] == "spread_sweep"].pivot(
        index="strategy", columns="spread_bps", values="ann_return"
    )
    focus_ret = focus_ret.reindex(columns=SPREADS)
    focus_ret = focus_ret.sort_values(15.0, ascending=False)
    log.info("\n%s", (focus_ret * 100).round(2).to_string())

    # Break-even analysis
    log.info("\n" + "=" * 90)
    log.info("CPPI-prod vs baseline — break-even spread")
    log.info("=" * 90)
    cppi_row = focus.loc["cppi_85_667_PROD"] if "cppi_85_667_PROD" in focus.index else None
    base_row = focus.loc["full_exposure"] if "full_exposure" in focus.index else None
    if cppi_row is not None and base_row is not None:
        for s in SPREADS:
            diff = cppi_row[s] - base_row[s]
            flag = "CPPI wins" if diff > 0 else "baseline wins"
            log.info("  %2g bps: CPPI Calmar=%.3f  baseline Calmar=%.3f  Δ=%+.3f  → %s",
                     s, cppi_row[s], base_row[s], diff, flag)

    return 0


if __name__ == "__main__":
    sys.exit(main())
