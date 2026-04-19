#!/usr/bin/env python3
"""
cppi_grid_sweep.py
==================
2D grid search around the current production CPPI config (85/6.67) to map
the local optimum on the (floor_pct, multiplier) surface.

Two sweep modes:
  1. "peak_lock": holds full (==1.0) exposure at peak by tying multiplier
     to floor as `mult = peak_target / (1 - floor)`. Sweeps floor and the
     implicit "peak_target" (1.0, 1.25, 1.5). Isolates the *shape* of the
     CPPI curve vs. the *speed* of cuts.
  2. "free": independent floor × multiplier grid — lets exposure at peak
     vary (some configs cap well below 1.0 on peak, burning cash drag).

Uses the existing backtest engine from backtest_capital_allocation.py.

Usage:
  python scripts/evaluation/cppi_grid_sweep.py --db data/kairos.duckdb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import duckdb
import pandas as pd

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluation.backtest_capital_allocation import (  # noqa: E402
    annualize, calmar, max_drawdown, run_backtest,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("cppi_grid")


def build_grid(mode: str) -> dict:
    strategies = {
        "baseline_full": {
            "fn": "full",
            "params": {},
            "description": "Baseline — 100% exposure always",
        },
    }
    if mode == "peak_lock":
        # Hold exposure-at-peak fixed across a range of floors
        floors = [0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.92]
        peak_targets = [1.00, 1.25, 1.50]
        for f in floors:
            for pt in peak_targets:
                mult = pt / (1.0 - f)
                name = f"cppi_{int(round(f*100))}_{mult:.2f}_pk{int(pt*100)}"
                strategies[name] = {
                    "fn": "cppi",
                    "params": {"floor_pct": f, "multiplier": mult},
                    "description": (
                        f"floor={f:.2f} mult={mult:.2f} (peak_target={pt:.2f})"
                    ),
                }
    elif mode == "free":
        floors = [0.78, 0.82, 0.85, 0.88, 0.90]
        multipliers = [4.0, 5.0, 6.67, 8.0, 10.0]
        for f in floors:
            for m in multipliers:
                name = f"cppi_{int(round(f*100))}_{m:.2f}"
                strategies[name] = {
                    "fn": "cppi",
                    "params": {"floor_pct": f, "multiplier": m},
                    "description": f"floor={f:.2f} mult={m:.2f}",
                }
    else:
        raise ValueError(mode)

    # Pin the current production config so it's always in the output
    strategies["cppi_85_667_PROD"] = {
        "fn": "cppi",
        "params": {"floor_pct": 0.85, "multiplier": 6.67},
        "description": "PRODUCTION: floor=0.85 mult=6.67 (peak=1.0)",
    }
    return strategies


def main() -> int:
    p = argparse.ArgumentParser(description="CPPI 2D grid sweep")
    p.add_argument("--db", required=True)
    p.add_argument("--alpha-column", default="alpha_ml_v2_tuned_clf")
    p.add_argument("--target-column", default="ret_5d_f")
    p.add_argument("--start-date", default="2015-01-01")
    p.add_argument("--end-date", default="2025-11-01")
    p.add_argument("--target-vol", type=float, default=0.25)
    p.add_argument("--mode", choices=("peak_lock", "free"), default="peak_lock")
    p.add_argument("--output-dir", default="outputs/evaluation/cppi_grid")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log.info("Loading panel + regime...")
    con = duckdb.connect(args.db, read_only=True)
    df = con.execute(f"""
        SELECT fm.ticker, fm.date,
               fm.{args.alpha_column} AS alpha,
               fm.{args.target_column} AS target,
               fm.vol_blend, fm.adv_20,
               t.sector
        FROM feat_matrix_v2 fm
        LEFT JOIN (SELECT DISTINCT ticker, sector FROM tickers
                   WHERE sector IS NOT NULL AND ticker != 'N/A') t USING (ticker)
        WHERE fm.{args.alpha_column} IS NOT NULL
          AND fm.{args.target_column} IS NOT NULL
          AND fm.vol_blend IS NOT NULL
          AND fm.adv_20 IS NOT NULL
          AND fm.date BETWEEN DATE '{args.start_date}' AND DATE '{args.end_date}'
        ORDER BY fm.date, fm.ticker
    """).fetchdf()
    regime_df = con.execute("SELECT date, vol_regime FROM regime_history_academic").fetchdf()
    con.close()
    log.info("  %d panel rows, %d regime rows", len(df), len(regime_df))

    strategies = build_grid(args.mode)
    log.info("Evaluating %d configs (mode=%s)", len(strategies), args.mode)

    results = []
    for name, cfg in strategies.items():
        bt = run_backtest(df.copy(), regime_df,
                          strategy_name=name, strategy_config=cfg,
                          target_vol=args.target_vol)
        if bt.empty:
            continue
        ann_ret, ann_vol, sharpe = annualize(bt["port_ret"], 5)
        eq = (1 + bt["port_ret"]).cumprod()
        mdd = max_drawdown(eq)
        cal = calmar(ann_ret, mdd)
        row = {
            "strategy": name,
            "floor": cfg["params"].get("floor_pct"),
            "multiplier": cfg["params"].get("multiplier"),
            "peak_exposure": (cfg["params"].get("multiplier", 1.0)
                              * (1.0 - cfg["params"].get("floor_pct", 0.0)))
                             if cfg["fn"] == "cppi" else 1.0,
            "sharpe": round(sharpe, 4),
            "ann_return": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4),
            "max_drawdown": round(mdd, 4),
            "calmar": round(cal, 3),
            "total_return": round(float(eq.iloc[-1] - 1), 4),
            "avg_allocation": round(float(bt["allocation"].mean()), 3),
            "min_allocation": round(float(bt["allocation"].min()), 3),
            "description": cfg["description"],
        }
        results.append(row)
        log.info("  %-32s  Sharpe=%+.3f  CAGR=%+.1f%%  DD=%+.1f%%  Calmar=%+.2f  AvgAlloc=%.0f%%",
                 name, sharpe, ann_ret * 100, mdd * 100, cal,
                 row["avg_allocation"] * 100)

    rdf = pd.DataFrame(results).sort_values("calmar", ascending=False)
    out_csv = out / f"cppi_grid_{args.mode}.csv"
    rdf.to_csv(out_csv, index=False)

    print("\n" + "=" * 120)
    print(f"CPPI {args.mode.upper()} GRID — sorted by Calmar")
    print("=" * 120)
    print(f"{'strategy':<36} {'floor':>6} {'mult':>6} {'pk_exp':>7} "
          f"{'Sharpe':>7} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'AvgAllc':>7}")
    print("-" * 120)
    for _, r in rdf.iterrows():
        fl = f"{r['floor']:.2f}" if r['floor'] is not None else "-"
        ml = f"{r['multiplier']:.2f}" if r['multiplier'] is not None else "-"
        pk = f"{r['peak_exposure']:.2f}" if r['peak_exposure'] is not None else "-"
        print(f"{r['strategy']:<36} {fl:>6} {ml:>6} {pk:>7} "
              f"{r['sharpe']:+.3f}  {r['ann_return']*100:+.1f}%  "
              f"{r['max_drawdown']*100:+.1f}%  {r['calmar']:+.2f}  "
              f"{r['avg_allocation']*100:.0f}%")

    prod = rdf[rdf["strategy"] == "cppi_85_667_PROD"].iloc[0]
    best = rdf.iloc[0]
    print(f"\nPRODUCTION: {prod['strategy']}  Calmar={prod['calmar']:.2f}  "
          f"Sharpe={prod['sharpe']:.3f}  CAGR={prod['ann_return']*100:.1f}%  "
          f"DD={prod['max_drawdown']*100:.1f}%")
    print(f"BEST:       {best['strategy']}  Calmar={best['calmar']:.2f}  "
          f"Sharpe={best['sharpe']:.3f}  CAGR={best['ann_return']*100:.1f}%  "
          f"DD={best['max_drawdown']*100:.1f}%")
    print(f"Δ vs PROD:  Calmar={best['calmar'] - prod['calmar']:+.2f}  "
          f"Sharpe={best['sharpe'] - prod['sharpe']:+.3f}  "
          f"CAGR={(best['ann_return'] - prod['ann_return'])*100:+.2f}pp  "
          f"DD={(best['max_drawdown'] - prod['max_drawdown'])*100:+.2f}pp")

    with open(out / f"cppi_grid_{args.mode}_summary.json", "w") as f:
        json.dump({"mode": args.mode, "results": results}, f, indent=2, default=str)
    log.info("Wrote %s", out_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
