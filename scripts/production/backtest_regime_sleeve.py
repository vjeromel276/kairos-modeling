#!/usr/bin/env python3
"""
backtest_regime_sleeve.py
=========================
Simple historical backtest comparing a baseline long-only top-N strategy
against the same strategy with a regime-aware cash sleeve applied
(scales equity exposure by the current regime_history_academic label).

At each weekly rebalance (Friday):
  - Rank universe by alpha signal
  - Top N (default 75) → equal weight inside the equity sleeve
  - Hold 5 trading days, collect forward return
  - Baseline: 100% invested in equity sleeve
  - Sleeve:   exposure[regime] of capital in equity sleeve, rest cash (0% rate)

Outputs:
  - cumulative return, Sharpe, max drawdown, Calmar for each variant
  - per-year returns
  - exposure time-series statistics
  - CSV and JSON summary in scripts/ml/outputs/

Usage:
  python scripts/production/backtest_regime_sleeve.py \\
      --db data/kairos.duckdb \\
      --start 2015-01-01 --end 2025-12-31 \\
      --alpha alpha_ml_v2_tuned_clf \\
      --top-n 75
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

from scripts.production.generate_rebalance import (  # noqa: E402
    REGIME_EXPOSURE as DEFAULT_REGIME_EXPOSURE,
    REGIME_EXPOSURE_DEFAULT,
)

# Alternative presets for comparison. The one in generate_rebalance.py is the
# "default" below — everything else is sweep candidates.
PRESETS = {
    "default": DEFAULT_REGIME_EXPOSURE,
    "moderate": {
        "low_vol_bull":       1.00,
        "normal_vol_bull":    1.00,
        "high_vol_bull":      1.00,
        "low_vol_neutral":    1.00,
        "normal_vol_neutral": 0.95,
        "high_vol_neutral":   0.85,
        "low_vol_bear":       0.90,
        "normal_vol_bear":    0.80,
        "high_vol_bear":      0.60,
    },
    "light": {
        "low_vol_bull":       1.00,
        "normal_vol_bull":    1.00,
        "high_vol_bull":      1.00,
        "low_vol_neutral":    1.00,
        "normal_vol_neutral": 1.00,
        "high_vol_neutral":   0.90,
        "low_vol_bear":       1.00,
        "normal_vol_bear":    0.90,
        "high_vol_bear":      0.75,
    },
    "bear_only": {
        # only trim in explicit bear regimes; ignore vol
        "low_vol_bull":       1.00,
        "normal_vol_bull":    1.00,
        "high_vol_bull":      1.00,
        "low_vol_neutral":    1.00,
        "normal_vol_neutral": 1.00,
        "high_vol_neutral":   1.00,
        "low_vol_bear":       0.60,
        "normal_vol_bear":    0.60,
        "high_vol_bear":      0.50,
    },
}
REGIME_EXPOSURE = PRESETS["default"]  # overwritten at runtime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("backtest_regime_sleeve")


def load_panel(con, start, end, alpha_col, min_adv):
    """Load panel with alpha, 5d forward return, regime, and ADV filter."""
    log.info("Loading panel (%s..%s, alpha=%s)", start, end, alpha_col)
    q = f"""
        SELECT t.ticker, t.date,
               m.{alpha_col}       AS alpha,
               m.adv_20            AS adv,
               t.ret_5d_f          AS ret_5d,
               r.regime            AS regime
        FROM feat_targets t
        JOIN feat_matrix_v2 m ON m.ticker = t.ticker AND m.date = t.date
        LEFT JOIN regime_history_academic r ON r.date = t.date
        WHERE t.date BETWEEN DATE '{start}' AND DATE '{end}'
          AND t.ret_5d_f IS NOT NULL
          AND m.{alpha_col} IS NOT NULL
          AND m.adv_20 IS NOT NULL
          AND m.adv_20 >= {min_adv}
    """
    df = con.execute(q).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    log.info("  %d rows, %d dates, %d tickers",
             len(df), df["date"].nunique(), df["ticker"].nunique())
    return df


def fridays_in_range(dates: pd.Series) -> list[pd.Timestamp]:
    """Last trading day per ISO week in the panel (approximates Friday rebalance)."""
    unique = pd.to_datetime(sorted(dates.unique()))
    df = pd.DataFrame({"date": unique})
    df["iso_year"] = df["date"].dt.isocalendar().year
    df["iso_week"] = df["date"].dt.isocalendar().week
    last = df.groupby(["iso_year", "iso_week"])["date"].max()
    return last.sort_values().tolist()


def drawdown(cum: np.ndarray) -> np.ndarray:
    peak = np.maximum.accumulate(cum)
    return cum / peak - 1.0


def summarize(returns: pd.Series, annualize_periods: float = 52) -> dict:
    """Portfolio stats for a per-rebalance return series (5d returns, ~52 per year)."""
    r = returns.dropna().to_numpy()
    if len(r) == 0:
        return {"n": 0}
    cum = np.cumprod(1 + r)
    total = cum[-1] - 1
    n_years = len(r) / annualize_periods
    cagr = cum[-1] ** (1 / n_years) - 1 if n_years > 0 else float("nan")
    mean_r = r.mean() * annualize_periods
    vol = r.std(ddof=1) * np.sqrt(annualize_periods)
    sharpe = mean_r / vol if vol > 0 else float("nan")
    dd = drawdown(cum)
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else float("nan")
    return {
        "n_periods": int(len(r)),
        "total_return": float(total),
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "calmar": float(calmar),
        "hit_rate": float((r > 0).mean()),
        "mean_period_return_bps": float(r.mean() * 1e4),
    }


def backtest(panel: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """For each Friday, take top-N by alpha, equal-weight, 5d forward return."""
    fridays = fridays_in_range(panel["date"])
    rows = []
    for d in fridays:
        sub = panel[panel["date"] == d]
        if len(sub) < top_n:
            continue
        top = sub.nlargest(top_n, "alpha")
        eq_return = top["ret_5d"].mean()  # equal weight across top N
        regime = sub["regime"].iloc[0]  # same for all rows on a given date
        exposure = REGIME_EXPOSURE.get(regime, REGIME_EXPOSURE_DEFAULT) \
            if pd.notna(regime) else REGIME_EXPOSURE_DEFAULT
        rows.append({
            "date": d,
            "regime": regime,
            "equity_return_5d": eq_return,
            "baseline_return_5d": eq_return,             # 100% invested
            "sleeve_exposure": exposure,
            "sleeve_return_5d": eq_return * exposure,     # cash portion earns 0
            "n_picks": top_n,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def per_year_stats(bt: pd.DataFrame, ret_col: str) -> pd.DataFrame:
    bt = bt.copy()
    bt["year"] = bt["date"].dt.year
    out = []
    for yr, g in bt.groupby("year"):
        s = summarize(g[ret_col])
        s["year"] = int(yr)
        out.append(s)
    return pd.DataFrame(out)


def main() -> int:
    p = argparse.ArgumentParser(description="Backtest: regime sleeve vs baseline")
    p.add_argument("--db", required=True, type=Path)
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--alpha", default="alpha_ml_v2_tuned_clf")
    p.add_argument("--top-n", type=int, default=75)
    p.add_argument("--min-adv", type=int, default=2_000_000)
    p.add_argument("--output-dir", type=Path, default=Path("scripts/ml/outputs"))
    p.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all",
                   help="Which exposure preset to evaluate (or 'all' to sweep)")
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(args.db), read_only=True)
    panel = load_panel(con, args.start, args.end, args.alpha, args.min_adv)
    con.close()

    # Which presets to evaluate
    presets_to_run = list(PRESETS) if args.preset == "all" else [args.preset]

    global REGIME_EXPOSURE
    # Compute baseline once using default preset for regime labelling (exposure
    # doesn't matter for the baseline_return_5d column — always 100%).
    REGIME_EXPOSURE = PRESETS[presets_to_run[0]]
    bt = backtest(panel, args.top_n)
    log.info("Backtest rows: %d (weekly rebalances %s → %s)",
             len(bt), bt["date"].min().date(), bt["date"].max().date())

    # Regime exposure distribution
    reg_dist = bt.groupby("regime").size().rename("n_rebalances").reset_index()
    reg_dist["pct"] = (reg_dist["n_rebalances"] / len(bt) * 100).round(1)
    reg_dist["exposure"] = reg_dist["regime"].map(
        lambda r: REGIME_EXPOSURE.get(r, REGIME_EXPOSURE_DEFAULT)
    )
    log.info("\nRegime distribution (across rebalance dates):")
    for _, r in reg_dist.sort_values("n_rebalances", ascending=False).iterrows():
        log.info("  %-22s  %4d  %5.1f%%  exposure=%.2f",
                 r["regime"], int(r["n_rebalances"]), r["pct"], r["exposure"])
    log.info("  avg sleeve exposure: %.3f", bt["sleeve_exposure"].mean())

    base_stats = summarize(bt["baseline_return_5d"])
    def fmt(d):
        return (f"Sharpe={d['sharpe']:+.2f}  CAGR={d['cagr']*100:+.2f}%  "
                f"maxDD={d['max_drawdown']*100:+.2f}%  Calmar={d['calmar']:+.2f}  "
                f"hit={d['hit_rate']:.2%}")
    log.info("\n" + "=" * 78)
    log.info("BASELINE: %s", fmt(base_stats))
    log.info("=" * 78)

    all_summaries = {"baseline": base_stats}
    for preset_name in presets_to_run:
        REGIME_EXPOSURE = PRESETS[preset_name]
        bt2 = backtest(panel, args.top_n)
        sleeve_stats = summarize(bt2["sleeve_return_5d"])
        avg_expo = bt2["sleeve_exposure"].mean()
        log.info("PRESET=%-10s (avg expo=%.2f)  %s",
                 preset_name, avg_expo, fmt(sleeve_stats))
        log.info("  Δ CAGR=%+.2fpp  ΔSharpe=%+.2f  ΔmaxDD=%+.2fpp  ΔCalmar=%+.2f",
                 (sleeve_stats["cagr"] - base_stats["cagr"]) * 100,
                 sleeve_stats["sharpe"] - base_stats["sharpe"],
                 (sleeve_stats["max_drawdown"] - base_stats["max_drawdown"]) * 100,
                 sleeve_stats["calmar"] - base_stats["calmar"])
        all_summaries[preset_name] = {
            "stats": sleeve_stats,
            "avg_exposure": float(avg_expo),
            "map": REGIME_EXPOSURE,
        }

    # Per-year for default preset (write default to CSV for inspection)
    REGIME_EXPOSURE = PRESETS[presets_to_run[0]]
    bt0 = backtest(panel, args.top_n)
    bt0.to_csv(args.output_dir / "backtest_regime_sleeve.csv", index=False)

    with open(args.output_dir / "backtest_regime_sleeve_summary.json", "w") as f:
        json.dump({
            "alpha_column": args.alpha,
            "date_range": [args.start, args.end],
            "top_n": args.top_n,
            "n_rebalances": len(bt),
            "regime_distribution": reg_dist.to_dict(orient="records"),
            "summaries": all_summaries,
        }, f, indent=2, default=str)
    log.info("\nWrote %s/backtest_regime_sleeve{.csv,_summary.json}", args.output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
