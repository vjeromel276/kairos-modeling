#!/usr/bin/env python3
"""
build_full_feature_matrix.py — FINAL ACADEMIC VERSION

Uses a temp table + iterative LEFT JOIN strategy
to avoid deep nested SELECT statements that break DuckDB's SQL parser.

Only includes:
    feat_price_action
    feat_price_shape
    feat_stat
    feat_trend
    feat_volume_volatility
    feat_targets
"""

import duckdb
import pandas as pd
import argparse
from pathlib import Path

ACADEMIC_TABLES = [
    "feat_price_action",
    "feat_price_shape",
    "feat_stat",
    "feat_trend",
    "feat_volume_volatility",
    "feat_targets",
    "feat_composite_academic"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--universe", required=True)
    args = parser.parse_args()

    con = duckdb.connect(args.db)

    print(f"📂 Loading universe CSV: {args.universe}")
    universe_df = pd.read_csv(args.universe)
    con.register("universe_df", universe_df)

    # ------------------------------------------------------------------
    # Validate required tables
    # ------------------------------------------------------------------
    existing = set(con.execute("SHOW TABLES").fetchdf()["name"].tolist())
    for t in ACADEMIC_TABLES:
        if t not in existing:
            raise RuntimeError(f"❌ Missing academic table: {t}")

    print("\n🔍 Using academic feature tables:")
    for t in ACADEMIC_TABLES:
        print(f"   • {t}")

    # ------------------------------------------------------------------
    # STEP 1: Create base temp table
    # ------------------------------------------------------------------
    print("\n🔧 Building temp base academic table...")

    con.execute("DROP TABLE IF EXISTS __feat_base")

    con.execute("""
        CREATE TABLE __feat_base AS
        SELECT
            s.ticker,
            s.date
        FROM sep_base_academic s
        INNER JOIN universe_df u USING (ticker)
        ORDER BY s.ticker, s.date
    """)

    # Confirm row count
    rows = con.execute("SELECT COUNT(*) FROM __feat_base").fetchone()[0]
    print(f"✔ Base contains {rows:,} rows")

    # ------------------------------------------------------------------
    # STEP 2: Iteratively LEFT JOIN each academic feature table
    # ------------------------------------------------------------------
    print("\n🔧 Iteratively joining feature tables...")

    for t in ACADEMIC_TABLES:

        print(f"   → Joining {t} ...")

        # Get column names, excluding ticker/date
        cols = con.execute(f"PRAGMA table_info('{t}')").fetchdf()["name"].tolist()
        feat_cols = [c for c in cols if c not in ("ticker", "date")]

        # Build left join
        select_cols = ", ".join(feat_cols) if feat_cols else ""

        con.execute(f"""
            CREATE OR REPLACE TABLE __feat_base AS
            SELECT
                b.*,
                {', '.join([f'f.{c}' for c in feat_cols])}
            FROM __feat_base b
            LEFT JOIN {t} f
            USING (ticker, date)
        """)

    # ------------------------------------------------------------------
    # STEP 3: Save to feat_matrix + parquet snapshot
    # ------------------------------------------------------------------
    print("\n💾 Saving final matrix...")

    con.execute("DROP TABLE IF EXISTS feat_matrix")
    con.execute("CREATE TABLE feat_matrix AS SELECT * FROM __feat_base")

    out_path = Path(f"scripts/feature_matrices/{args.date}_academic_feature_matrix.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = con.execute("SELECT * FROM feat_matrix").fetchdf()
    df.to_parquet(out_path, index=False)

    print(f"✔ Saved {len(df):,} rows × {df.shape[1]} columns to feat_matrix")
    print(f"✔ Parquet snapshot: {out_path}")

    # Cleanup
    con.execute("DROP TABLE IF EXISTS __feat_base")
    con.close()


if __name__ == "__main__":
    main()
