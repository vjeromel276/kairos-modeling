#!/usr/bin/env python3
# table_analysis.py — compute coverage curve without loading all rows into pandas

import argparse
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Coverage curve: for each year, count tickers with ≥threshold pre-history days before Jan 1 of that year."
    )
    p.add_argument("--db-path", default="data/kairos.duckdb")
    p.add_argument("--output-coverage-csv", default="coverage_by_year.csv")
    p.add_argument("--output-plot", default="coverage_over_time.png")
    p.add_argument("--threshold", type=int, default=252 + 21)
    return p.parse_args()


def main():
    args = parse_args()

    conn = duckdb.connect(database=args.db_path, read_only=True)
    print(f"Connected to DuckDB at {args.db_path}")

    # Pull min/max dates once
    min_max = conn.execute(
        "SELECT MIN(CAST(date AS DATE)) AS min_d, MAX(CAST(date AS DATE)) AS max_d FROM sep_base_common"
    ).fetchone()
    min_d, max_d = pd.to_datetime(min_max[0]).date(), pd.to_datetime(min_max[1]).date() # type: ignore
    print(f"Analysis years: {min_d.year} to {max_d.year}")

    years = list(range(min_d.year, max_d.year + 1))
    rows = []
    for yr in years:
        cutoff = f"{yr}-01-01"
        # Eligible tickers: those appearing in sep_base_common on/before cutoff
        eligible_sql = f"""
            WITH eligible AS (
              SELECT ticker
              FROM sep_base_common
              GROUP BY ticker
              HAVING MIN(CAST(date AS DATE)) <= DATE '{cutoff}'
            ), pre AS (
              SELECT b.ticker, COUNT(*) AS pre_days
              FROM sep_base b
              JOIN eligible e USING (ticker)
              WHERE CAST(b.date AS DATE) < DATE '{cutoff}'
              GROUP BY b.ticker
            )
            SELECT
              (SELECT COUNT(*) FROM eligible)                    AS tickers_eligible,
              (SELECT COUNT(*) FROM pre WHERE pre_days >= {args.threshold}) AS tickers_available
        """
        tickers_eligible, tickers_available = conn.execute(eligible_sql).fetchone() # type: ignore
        rows.append({
            "year": yr,
            "tickers_eligible": int(tickers_eligible or 0),
            "tickers_available": int(tickers_available or 0)
        })

    conn.close()

    cov = pd.DataFrame(rows)
    cov["coverage_ratio"] = cov.apply(
        lambda r: (r["tickers_available"] / r["tickers_eligible"]) if r["tickers_eligible"] else 0.0,
        axis=1,
    )

    os.makedirs(os.path.dirname(args.output_coverage_csv) or ".", exist_ok=True)
    cov.to_csv(args.output_coverage_csv, index=False)
    print(f"Saved coverage summary → {args.output_coverage_csv}")

    # Plot
    plt.figure()
    plt.plot(cov["year"], cov["tickers_available"], marker='o')
    plt.xlabel("Year")
    plt.ylabel(f"Tickers with ≥{args.threshold} pre-history days")
    plt.title("Ticker Coverage Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.output_plot)
    print(f"Saved coverage plot → {args.output_plot}")


if __name__ == "__main__":
    main()
