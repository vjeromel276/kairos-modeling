#!/usr/bin/env python3
"""
Build ownership-based features from institutional and insider data (sf3a + sf3b)

Features:
- institutional_ownership_pct
- institutional_churn_1q
- insider_buy_sell_ratio
- insider_net_activity_1m
"""

import duckdb
import pandas as pd
import argparse

def compute_ownership_features(con):
    # Load sf3b (institutional ownership) quarterly
    inst = con.execute("""
        SELECT ticker, date, sharesheld, sharesout
        FROM sf3b
        WHERE sharesheld IS NOT NULL AND sharesout IS NOT NULL AND sharesout > 0
        ORDER BY ticker, date
    """).fetchdf()

    inst["inst_own_pct"] = inst["sharesheld"] / inst["sharesout"]
    inst["inst_churn_1q"] = inst.groupby("ticker")["inst_own_pct"].diff()

    inst = inst[["ticker", "date", "inst_own_pct", "inst_churn_1q"]]

    # Load sf3a (insider trades), aggregated monthly
    insider = con.execute("""
        SELECT ticker, date, shares, type
        FROM sf3a
        WHERE shares IS NOT NULL AND type IN ('Buy', 'Sell')
        ORDER BY ticker, date
    """).fetchdf()

    insider["shares"] = insider["shares"].astype(float)
    insider["buy_shares"] = insider["shares"].where(insider["type"] == "Buy", 0)
    insider["sell_shares"] = insider["shares"].where(insider["type"] == "Sell", 0)

    monthly = (
        insider.groupby(["ticker", "date"])
        .agg({
            "buy_shares": "sum",
            "sell_shares": "sum"
        })
        .reset_index()
    )

    monthly["insider_net"] = monthly["buy_shares"] - monthly["sell_shares"]
    monthly["buy_sell_ratio"] = monthly["buy_shares"] / (monthly["sell_shares"] + 1e-5)

    monthly = monthly.rename(columns={"date": "insider_date"})

    # Join institutional + insider (fuzzy join by month end)
    merged = pd.merge_asof(
        inst.sort_values("date"),
        monthly.sort_values("insider_date"),
        by="ticker",
        left_on="date",
        right_on="insider_date",
        direction="backward",
        tolerance=pd.Timedelta(days=30)
    )

    df = merged.dropna()

    return df[[
        "ticker", "date",
        "inst_own_pct", "inst_churn_1q",
        "buy_sell_ratio", "insider_net"
    ]]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    args = parser.parse_args()

    con = duckdb.connect(args.db)
    con.execute("DROP TABLE IF EXISTS feat_ownership")

    df_feat = compute_ownership_features(con)
    con.execute("CREATE TABLE feat_ownership AS SELECT * FROM df_feat")

    print(f"✅ Saved {len(df_feat):,} rows to feat_ownership in {args.db}")

if __name__ == "__main__":
    main()
