#!/usr/bin/env python3
"""
backtest_regime_switching.py

Phase 8D — Regime-Aware Portfolio Switching

Combines:
    - Long-only engine (8A): backtest_results_longonly_r4
    - Long-short optimized engine (8C): backtest_results_ls_opt
with:
    - regime_history_academic.regime  (from regime_detector_academic.py)

Regimes look like: 'low_vol_bull', 'high_vol_bear', 'normal_vol_neutral', etc.

Mapping (meta-regime → engine weights):
    - Bullish regimes   → mostly long-only
    - Bearish regimes   → mostly long-short optimized
    - Neutral regimes   → blended

Usage:

    python scripts/backtesting/backtest_regime_switching.py \
      --db data/kairos.duckdb \
      --start-date 2015-01-01 \
      --end-date 2025-11-28
"""

import argparse
import logging
from typing import Tuple, Optional

import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


# =====================================================================
# Metrics helpers
# =====================================================================

def max_drawdown(eq: pd.Series) -> float:
    run_max = eq.cummax()
    dd = eq / run_max - 1.0
    return float(dd.min())


def annualize(returns: pd.Series, period_days: int = 5) -> Tuple[float, float, float]:
    if returns.empty:
        return 0.0, 0.0, 0.0
    af = 252.0 / period_days
    gross = (1.0 + returns).prod()
    avg = gross ** (1.0 / len(returns)) - 1.0
    ann_ret = (1.0 + avg) ** af - 1.0
    ann_vol = returns.std() * np.sqrt(af)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    return float(ann_ret), float(ann_vol), float(sharpe)


# =====================================================================
# Regime mapping
# =====================================================================

def regime_to_weights(regime: str) -> Tuple[float, float]:
    """
    Map raw regime string (e.g. 'low_vol_bull') to (w_longonly, w_ls_opt).

    Intuition:
        - Bull markets  → long-only dominates
        - Bear markets  → optimized long-short dominates
        - Neutral       → blend

    You can tweak these later if desired.
    """
    if not isinstance(regime, str) or "_" not in regime:
        return 0.5, 0.5

    vol_regime, trend_regime = regime.split("_", 1)

    # Bullish trend regimes
    if trend_regime == "bull":
        if vol_regime in ("low_vol", "normal_vol"):
            return 1.0, 0.0       # pure long-only
        else:  # high_vol_bull
            return 0.7, 0.3       # mostly long-only, some LS hedge

    # Bearish trend regimes
    if trend_regime == "bear":
        if vol_regime == "high_vol":
            return 0.0, 1.0       # full LS hedge
        else:
            return 0.3, 0.7       # mostly LS, keep some long-only optionality

    # Neutral / choppy trend regimes
    if trend_regime == "neutral":
        if vol_regime == "low_vol":
            return 0.6, 0.4
        elif vol_regime == "normal_vol":
            return 0.5, 0.5
        else:  # high_vol_neutral
            return 0.3, 0.7

    # Fallback
    return 0.5, 0.5


# =====================================================================
# Data load
# =====================================================================

def load_inputs(
    con: duckdb.DuckDBPyConnection,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load:
        - regime_history_academic
        - backtest_results_longonly_r4
        - backtest_results_ls_opt
    and align by date.
    """
    logger.info("Loading regime history...")
    where_reg = []
    if start:
        where_reg.append(f"date >= DATE '{start}'")
    if end:
        where_reg.append(f"date <= DATE '{end}'")
    where_reg_sql = " AND ".join(where_reg) if where_reg else "TRUE"

    regimes = con.execute(f"""
        SELECT date, regime
        FROM regime_history_academic
        WHERE {where_reg_sql}
        ORDER BY date
    """).fetchdf()

    logger.info(f"Loaded {len(regimes):,} regime rows.")

    logger.info("Loading long-only 8A results (backtest_results_longonly_r4)...")
    lo = con.execute("""
        SELECT date, port_ret, bench_ret, active_ret
        FROM backtest_results_longonly_r4
        ORDER BY date
    """).fetchdf()

    logger.info(f"Loaded {len(lo):,} long-only rows.")

    logger.info("Loading optimized LS 8C results (backtest_results_ls_opt)...")
    ls = con.execute("""
        SELECT date, port_ret, bench_ret, active_ret
        FROM backtest_results_ls_opt
        ORDER BY date
    """).fetchdf()

    logger.info(f"Loaded {len(ls):,} LS-opt rows.")

    return regimes, lo, ls


# =====================================================================
# Core regime-switching engine
# =====================================================================

def build_regime_switched_portfolio(
    regimes: pd.DataFrame,
    lo: pd.DataFrame,
    ls: pd.DataFrame,
    rebalance_every: int = 5,
) -> pd.DataFrame:
    """
    Combine long-only and LS-opt returns using regime-based weights.
    """

    # Merge returns and regimes on date (inner join so we only use common dates)
    logger.info("Merging regime + engine returns by date...")
    df = (
        regimes.merge(lo, on="date", how="inner", suffixes=("", "_lo"))
               .merge(ls, on="date", how="inner", suffixes=("_lo", "_ls"))
    )

    if df.empty:
        logger.warning("No overlapping dates between regime and backtest results.")
        return df

    df = df.sort_values("date").reset_index(drop=True)

    # Rename for clarity
    df.rename(
        columns={
            "port_ret_lo": "port_ret_long",
            "bench_ret_lo": "bench_ret_long",
            "active_ret_lo": "active_ret_long",
            "port_ret_ls": "port_ret_ls",
            "bench_ret_ls": "bench_ret_ls",
            "active_ret_ls": "active_ret_ls",
        },
        inplace=True,
    )

    # Some versions of DuckDB might keep old names; handle gracefully
    if "port_ret" in df.columns and "port_ret_long" not in df.columns:
        df.rename(columns={"port_ret": "port_ret_long"}, inplace=True)
    if "port_ret_ls_opt" in df.columns and "port_ret_ls" not in df.columns:
        df.rename(columns={"port_ret_ls_opt": "port_ret_ls"}, inplace=True)

    # Apply regime → weights per date
    w_long_list = []
    w_ls_list = []
    for reg in df["regime"]:
        w_long, w_ls = regime_to_weights(reg)
        w_long_list.append(w_long)
        w_ls_list.append(w_ls)

    df["w_long"] = w_long_list
    df["w_ls"] = w_ls_list

    # Combined portfolio returns
    df["port_ret"] = df["w_long"] * df["port_ret_long"] + df["w_ls"] * df["port_ret_ls"]
    # Use long-only benchmark as "beta" benchmark
    df["bench_ret"] = df["bench_ret_long"]
    df["active_ret"] = df["port_ret"] - df["bench_ret"]

    return df


# =====================================================================
# Summary + per-regime diagnostics
# =====================================================================

def summarize(df: pd.DataFrame, rebalance_every: int, target_vol_info: str = "") -> None:
    if df.empty:
        print("No regime-switched results to summarize.")
        return

    p = df["port_ret"]
    b = df["bench_ret"]
    a = df["active_ret"]

    port_eq = (1.0 + p).cumprod()
    bench_eq = (1.0 + b).cumprod()

    port_total = float(port_eq.iloc[-1] - 1.0)
    bench_total = float(bench_eq.iloc[-1] - 1.0)

    port_ann, port_vol, port_sharpe = annualize(p, rebalance_every)
    bench_ann, bench_vol, bench_sharpe = annualize(b, rebalance_every)
    active_ann, active_vol, active_sharpe = annualize(a, rebalance_every)

    port_dd = max_drawdown(port_eq)
    bench_dd = max_drawdown(bench_eq)

    print("\n==================== REGIME-AWARE PORTFOLIO (PHASE 8D) ====================")
    if target_vol_info:
        print(f"{target_vol_info}")
    print("---------------------------------------------------------------------")
    print(f"PORTFOLIO Total Return:  {port_total:7.2%}")
    print(f"PORTFOLIO Annual Ret:    {port_ann:7.2%}")
    print(f"PORTFOLIO Annual Vol:    {port_vol:7.2%}")
    print(f"PORTFOLIO Sharpe:        {port_sharpe:7.2f}")
    print(f"PORTFOLIO Max DD:        {port_dd:7.2%}")
    print("---------------------------------------------------------------------")
    print(f"BENCHMARK Total:         {bench_total:7.2%}")
    print(f"BENCHMARK Annual Ret:    {bench_ann:7.2%}")
    print(f"BENCHMARK Annual Vol:    {bench_vol:7.2%}")
    print(f"BENCHMARK Sharpe:        {bench_sharpe:7.2f}")
    print(f"BENCHMARK Max DD:        {bench_dd:7.2%}")
    print("---------------------------------------------------------------------")
    print(f"ACTIVE Annual Ret:       {active_ann:7.2%}")
    print(f"ACTIVE Annual Vol:       {active_vol:7.2%}")
    print(f"ACTIVE Sharpe:           {active_sharpe:7.2f}")
    print("=====================================================================\n")

    # Per-regime diagnostics
    print("Per-regime performance (portfolio Sharpe by regime):")
    per_reg = []
    for reg, g in df.groupby("regime"):
        _, _, sh = annualize(g["port_ret"], rebalance_every)
        per_reg.append((reg, sh, len(g)))
    per_reg = sorted(per_reg, key=lambda x: x[1], reverse=True)
    for reg, sh, n in per_reg:
        print(f"  {reg:20s}  Sharpe={sh:6.2f}  periods={n}")


# =====================================================================
# CLI
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 8D Regime-Aware Portfolio Backtest.")
    parser.add_argument("--db", required=True)
    parser.add_argument("--rebalance-every", type=int, default=5)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    regimes, lo, ls = load_inputs(con, args.start_date, args.end_date)
    df = build_regime_switched_portfolio(regimes, lo, ls, rebalance_every=args.rebalance_every)

    summarize(df, rebalance_every=args.rebalance_every)

    # Optionally save to DuckDB
    con.execute("DROP TABLE IF EXISTS backtest_results_regime_switching")
    con.register("bt_reg", df)
    con.execute("CREATE TABLE backtest_results_regime_switching AS SELECT * FROM bt_reg")
    logger.info("Saved backtest_results_regime_switching to DuckDB.")

    con.close()


if __name__ == "__main__":
    main()
