#!/usr/bin/env python3
"""
Check which SF1 columns actually have data
"""

import argparse
import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("SF1 COLUMN DATA CHECK")
    print("=" * 80)

    # Get all columns
    cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'sf1'
        ORDER BY ordinal_position
    """).df()['column_name'].tolist()
    
    print(f"\nSF1 has {len(cols)} columns")
    print("\nChecking non-null counts for ARQ dimension...\n")
    
    print(f"{'Column':<25} {'Non-Null':>12} {'Total':>12} {'Coverage':>10}")
    print("-" * 65)
    
    total_arq = con.execute("SELECT COUNT(*) FROM sf1 WHERE dimension = 'ARQ'").fetchone()[0]
    
    # Key fundamental columns to check
    key_cols = [
        'roe', 'roa', 'roic', 'ros',  # Profitability
        'grossmargin', 'netmargin', 'ebitdamargin',  # Margins
        'pe', 'pb', 'ps', 'divyield',  # Valuation
        'de', 'currentratio', 'assetturnover',  # Efficiency
        'revenue', 'netinc', 'ncfo', 'assets',  # Raw financials
        'eps', 'bvps', 'fcfps',  # Per share
        'marketcap', 'ev', 'ebit', 'ebitda'  # Other
    ]
    
    for col in key_cols:
        if col in cols:
            try:
                non_null = con.execute(f"""
                    SELECT COUNT({col}) 
                    FROM sf1 
                    WHERE dimension = 'ARQ'
                """).fetchone()[0]
                coverage = 100 * non_null / total_arq if total_arq > 0 else 0
                status = "✓" if coverage > 50 else "⚠️" if coverage > 10 else "✗"
                print(f"{col:<25} {non_null:>12,} {total_arq:>12,} {coverage:>9.1f}% {status}")
            except Exception as e:
                print(f"{col:<25} ERROR: {str(e)[:30]}")
        else:
            print(f"{col:<25} NOT IN TABLE")

    # Check if ROE might be stored differently
    print("\n" + "=" * 80)
    print("CHECKING ALTERNATIVE ROE/ROA COLUMNS")
    print("=" * 80)
    
    # Search for any column with 'ro' in name
    ro_cols = [c for c in cols if 'ro' in c.lower()]
    print(f"\nColumns containing 'ro': {ro_cols}")
    
    for col in ro_cols:
        try:
            stats = con.execute(f"""
                SELECT 
                    COUNT({col}) as non_null,
                    AVG({col}) as avg_val,
                    MIN({col}) as min_val,
                    MAX({col}) as max_val
                FROM sf1 
                WHERE dimension = 'ARQ'
            """).df()
            print(f"\n{col}:")
            print(stats.to_string(index=False))
        except:
            pass

    # Sample a few rows with common financials
    print("\n" + "=" * 80)
    print("SAMPLE SF1 DATA (AAPL)")
    print("=" * 80)
    
    sample = con.execute("""
        SELECT 
            ticker, datekey, dimension,
            revenue, netinc, assets, equity,
            roe, roa, grossmargin, netmargin
        FROM sf1
        WHERE ticker = 'AAPL' 
          AND dimension = 'ARQ'
        ORDER BY datekey DESC
        LIMIT 5
    """).df()
    print(sample.to_string(index=False))

    # Check if we can COMPUTE ROE
    print("\n" + "=" * 80)
    print("CAN WE COMPUTE ROE FROM RAW DATA?")
    print("=" * 80)
    
    compute_check = con.execute("""
        SELECT 
            COUNT(*) as total,
            COUNT(netinc) as has_netinc,
            COUNT(equity) as has_equity,
            COUNT(CASE WHEN netinc IS NOT NULL AND equity IS NOT NULL AND equity != 0 
                       THEN 1 END) as can_compute_roe
        FROM sf1
        WHERE dimension = 'ARQ'
    """).df()
    print(compute_check.to_string(index=False))
    
    # Test computed ROE
    print("\nSample computed ROE (netinc/equity):")
    computed = con.execute("""
        SELECT 
            ticker, datekey,
            netinc, equity,
            netinc / NULLIF(equity, 0) as computed_roe
        FROM sf1
        WHERE dimension = 'ARQ'
          AND netinc IS NOT NULL 
          AND equity IS NOT NULL
          AND equity != 0
        ORDER BY datekey DESC
        LIMIT 10
    """).df()
    print(computed.to_string(index=False))

    con.close()

if __name__ == "__main__":
    main()