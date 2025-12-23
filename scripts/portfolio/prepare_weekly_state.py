#!/usr/bin/env python3
"""
prepare_weekly_state.py

Inspect database state and determine whether the system is ready
to run the weekly portfolio decision.

Key concepts:
- Market clock: what the market has traded (sep_base)
- Information clock: what information is available (fundamentals)
- Decision date = min(market_date, info_date)

Exit codes:
- 0 = READY
- 1 = NOT READY (blocking dependencies)
"""

import argparse
import sys
from datetime import datetime, date
import duckdb


DB_PATH = "data/kairos.duckdb"
UNIVERSE_CSV = "scripts/sep_dataset/feature_sets/option_b_universe.csv"
TARGET_HORIZON_DAYS = 5  # ret_5d_f


# -------------------------
# Helpers
# -------------------------

def parse_date(d: str) -> date:
    try:
        return datetime.strptime(d, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {d} (expected YYYY-MM-DD)")


def max_date(con, table: str):
    try:
        return con.execute(f"SELECT MAX(date) FROM {table}").fetchone()[0]
    except duckdb.CatalogException:
        return None


def resolve_last_market_date(con, as_of: date):
    row = con.execute(
        """
        SELECT MAX(date)
        FROM sep_base
        WHERE date <= ?
        """,
        [as_of],
    ).fetchone()
    return row[0]


def trading_dates_before(con, ref_date, n_days):
    rows = con.execute(
        """
        SELECT DISTINCT date
        FROM sep_base
        WHERE date < ?
        ORDER BY date DESC
        LIMIT ?
        """,
        [ref_date, n_days],
    ).fetchall()
    if len(rows) < n_days:
        return None
    return rows[-1][0]


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare weekly state (inspect-only).")
    parser.add_argument(
        "--as-of-date",
        required=True,
        help="Date up to which market data is expected (YYYY-MM-DD)",
    )
    args = parser.parse_args()

    try:
        as_of_date = parse_date(args.as_of_date)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    try:
        con = duckdb.connect(DB_PATH, read_only=True)
    except Exception as e:
        print(f"ERROR: Could not connect to DuckDB: {e}")
        sys.exit(1)

    # -------------------------
    # Resolve clocks
    # -------------------------

    market_date = resolve_last_market_date(con, as_of_date)
    if market_date is None:
        print(f"ERROR: No trading date found <= {as_of_date}")
        sys.exit(1)

    quality_date = max_date(con, "feat_quality_v2")
    v7_date = max_date(con, "feat_composite_v7")
    v33_date = max_date(con, "feat_composite_v33_regime")
    matrix_date = max_date(con, "feat_matrix_v2")
    targets_date = max_date(con, "feat_targets")

    decision_date = min(d for d in [market_date, v7_date] if d is not None)

    expected_targets_date = trading_dates_before(
        con, market_date, TARGET_HORIZON_DAYS
    )

    # -------------------------
    # Print header
    # -------------------------

    print("\nWEEKLY PREPARATION CHECK")
    print("========================")
    print(f"As-of date:       {as_of_date}")
    print(f"Market date:      {market_date}")
    print(f"Decision date:    {decision_date}")
    print()

    # -------------------------
    # Component status
    # -------------------------

    print("COMPONENT STATUS")
    print("----------------")

    def ok(label, d): print(f"{label:<30} OK     ({d})")
    def lag(label, d, note): print(f"{label:<30} OK     ({d})  [{note}]")
    def stale(label, d): print(f"{label:<30} STALE  ({d})")

    ok("sep_base", market_date)
    ok("sep_base_academic", max_date(con, "sep_base_academic"))

    for t in [
        "feat_price_action",
        "feat_price_shape",
        "feat_stat",
        "feat_trend",
        "feat_volume_volatility",
        "feat_adv",
        "feat_vol_sizing",
        "feat_beta",
    ]:
        d = max_date(con, t)
        if d == market_date:
            ok(t, d)
        else:
            stale(t, d)

    # Targets (horizon-lagged)
    if targets_date == expected_targets_date:
        lag("feat_targets", targets_date, "horizon-lagged")
    else:
        stale("feat_targets", targets_date)

    # Quality (fundamental-lagged)
    lag("feat_quality_v2", quality_date, "fundamental-lagged")

    # Composites
    if v33_date == market_date:
        ok("feat_composite_v33_regime", v33_date)
    else:
        stale("feat_composite_v33_regime", v33_date)

    if v7_date == decision_date:
        lag("feat_composite_v7", v7_date, "fundamental-lagged")
    else:
        stale("feat_composite_v7", v7_date)

    # Matrix must match decision date
    if matrix_date == decision_date:
        ok("feat_matrix_v2", matrix_date)
    else:
        stale("feat_matrix_v2", matrix_date)

    print()

    # -------------------------
    # Determine blocking issues
    # -------------------------

    blocking = []

    if v33_date != market_date:
        blocking.append("Rebuild feat_composite_v33_regime")

    if v7_date != decision_date:
        blocking.append("Rebuild feat_composite_v7")

    if matrix_date != decision_date:
        blocking.append("Rebuild feat_matrix_v2")

    # -------------------------
    # Output plan
    # -------------------------

    if not blocking:
        print("STATUS: READY FOR WEEKLY PORTFOLIO RUN\n")
        print("NEXT STEPS:")
        print("  python scripts/portfolio/run_weekly_regime.py")
        print("  python scripts/portfolio/run_weekly_primary.py")
        print("  python scripts/portfolio/run_weekly_hedge.py")
        print()
        sys.exit(0)

    print("BLOCKING ACTIONS REQUIRED")
    print("-------------------------")

    step = 1
    if "Rebuild feat_composite_v33_regime" in blocking:
        print(f"{step}. python scripts/features/build_composite_v33_regime.py --db {DB_PATH}")
        step += 1

    if "Rebuild feat_composite_v7" in blocking:
        print(f"{step}. python scripts/features/build_alpha_composite_v7.py --db {DB_PATH}")
        step += 1

    if "Rebuild feat_matrix_v2" in blocking:
        print(
            f"{step}. python scripts/build_feature_matrix_v2.py "
            f"--db {DB_PATH} --date {decision_date} --universe {UNIVERSE_CSV}"
        )
        step += 1

    print("\nAFTER COMPLETING THE ABOVE:")
    print("  python scripts/portfolio/run_weekly_regime.py")
    print("  python scripts/portfolio/run_weekly_primary.py")
    print("  python scripts/portfolio/run_weekly_hedge.py")
    print()

    sys.exit(1)


if __name__ == "__main__":
    main()
