#!/usr/bin/env python3
"""
Sharadar Data Audit & Alpha Factor Opportunity Assessment

Goal: Identify what data you have and which academically-proven 
alpha factors you could build.

High-IC factors from academic literature:
1. Earnings Quality / Accruals (-3-5% IC)
2. Earnings Momentum / SUE (2-4% IC)  
3. Short Interest (2-4% IC)
4. Insider Buying (1-3% IC)
5. Value (book/market, earnings yield) (1-2% IC)
6. Profitability (ROE, ROA, gross margins) (1-2% IC)
7. Investment (asset growth, capex) (-1-2% IC, negative is good)
8. Momentum (2-12 month) (2-4% IC)
9. Low Volatility (1-2% IC)
10. Quality (composite of profitability + safety) (2-3% IC)
"""

import argparse
import duckdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=str, default="data/kairos.duckdb")
    args = parser.parse_args()

    con = duckdb.connect(args.db, read_only=True)

    print("=" * 80)
    print("SHARADAR DATA AUDIT & ALPHA OPPORTUNITY ASSESSMENT")
    print("=" * 80)

    # =========================================================================
    # 1. LIST ALL TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("1. ALL TABLES IN DATABASE")
    print("=" * 80)
    
    tables = con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'main'
        ORDER BY table_name
    """).df()
    
    print(f"\nFound {len(tables)} tables:")
    for t in tables['table_name']:
        row_count = con.execute(f"SELECT COUNT(*) as cnt FROM {t}").fetchone()[0]
        print(f"  {t}: {row_count:,} rows")

    # =========================================================================
    # 2. CHECK FOR FUNDAMENTAL DATA (Sharadar SF1)
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. FUNDAMENTAL DATA CHECK (Sharadar SF1)")
    print("=" * 80)
    
    # Look for tables that might contain fundamentals
    fundamental_tables = ['sf1', 'fundamentals', 'sharadar_sf1', 'sep', 'daily_sep']
    
    for table in fundamental_tables:
        try:
            cols = con.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).df()
            if len(cols) > 0:
                print(f"\n✓ Found table: {table}")
                print(f"  Columns ({len(cols)}): {cols['column_name'].tolist()[:20]}...")
                
                # Sample data
                sample = con.execute(f"SELECT * FROM {table} LIMIT 3").df()
                print(f"  Sample:")
                print(sample.to_string())
        except:
            pass

    # =========================================================================
    # 3. CHECK FEAT_MATRIX FOR EXISTING FACTORS
    # =========================================================================
    print("\n" + "=" * 80)
    print("3. EXISTING FACTORS IN FEAT_MATRIX")
    print("=" * 80)
    
    fm_cols = con.execute("""
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name = 'feat_matrix'
        ORDER BY column_name
    """).df()
    
    print(f"\nTotal columns in feat_matrix: {len(fm_cols)}")
    
    # Categorize columns
    categories = {
        'price_momentum': [],
        'volume': [],
        'volatility': [],
        'value': [],
        'quality': [],
        'institutional': [],
        'alpha': [],
        'return': [],
        'other': []
    }
    
    for col in fm_cols['column_name']:
        col_lower = col.lower()
        if 'ret' in col_lower or 'mom' in col_lower:
            categories['price_momentum'].append(col)
        elif 'vol' in col_lower and 'volatil' not in col_lower:
            categories['volume'].append(col)
        elif 'vol' in col_lower or 'std' in col_lower or 'var' in col_lower:
            categories['volatility'].append(col)
        elif any(x in col_lower for x in ['pe', 'pb', 'ps', 'ev', 'yield', 'value', 'book', 'earn']):
            categories['value'].append(col)
        elif any(x in col_lower for x in ['roe', 'roa', 'margin', 'profit', 'quality']):
            categories['quality'].append(col)
        elif any(x in col_lower for x in ['inst', '13f', 'owner']):
            categories['institutional'].append(col)
        elif 'alpha' in col_lower:
            categories['alpha'].append(col)
        elif 'ret' in col_lower:
            categories['return'].append(col)
        else:
            categories['other'].append(col)
    
    for cat, cols in categories.items():
        if cols:
            print(f"\n{cat.upper()} ({len(cols)}):")
            print(f"  {cols[:10]}{'...' if len(cols) > 10 else ''}")

    # =========================================================================
    # 4. CHECK FOR VALUE FACTORS
    # =========================================================================
    print("\n" + "=" * 80)
    print("4. VALUE FACTOR DATA")
    print("=" * 80)
    
    # Check feat_value table
    try:
        value_cols = con.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'feat_value'
        """).df()
        print(f"\nfeat_value columns: {value_cols['column_name'].tolist()}")
        
        # Sample
        sample = con.execute("SELECT * FROM feat_value LIMIT 5").df()
        print("\nSample:")
        print(sample.to_string())
        
        # Check IC of value factors
        print("\nValue factor ICs:")
        for col in value_cols['column_name']:
            if col not in ['ticker', 'date']:
                try:
                    ic = con.execute(f"""
                        SELECT CORR(v.{col}, t.ret_5d_f) as ic
                        FROM feat_value v
                        JOIN feat_targets t ON v.ticker = t.ticker AND v.date = t.date
                        WHERE v.{col} IS NOT NULL AND t.ret_5d_f IS NOT NULL
                    """).fetchone()[0]
                    if ic is not None:
                        print(f"  {col}: IC = {ic:.4f}")
                except Exception as e:
                    pass
    except Exception as e:
        print(f"feat_value not found or error: {e}")

    # =========================================================================
    # 5. CHECK FOR INSTITUTIONAL DATA
    # =========================================================================
    print("\n" + "=" * 80)
    print("5. INSTITUTIONAL / 13F DATA")
    print("=" * 80)
    
    try:
        inst_cols = con.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'feat_institutional_academic'
        """).df()
        print(f"\nfeat_institutional_academic columns: {inst_cols['column_name'].tolist()}")
        
        # Check IC
        print("\nInstitutional factor ICs:")
        for col in inst_cols['column_name']:
            if col not in ['ticker', 'date']:
                try:
                    ic = con.execute(f"""
                        SELECT CORR(i.{col}, t.ret_5d_f) as ic
                        FROM feat_institutional_academic i
                        JOIN feat_targets t ON i.ticker = t.ticker AND i.date = t.date
                        WHERE i.{col} IS NOT NULL AND t.ret_5d_f IS NOT NULL
                    """).fetchone()[0]
                    if ic is not None:
                        print(f"  {col}: IC = {ic:.4f}")
                except:
                    pass
    except Exception as e:
        print(f"Error: {e}")

    # =========================================================================
    # 6. CHECK FOR RAW SHARADAR TABLES
    # =========================================================================
    print("\n" + "=" * 80)
    print("6. RAW SHARADAR TABLES")
    print("=" * 80)
    
    sharadar_tables = ['sep', 'sf1', 'sf3', 'daily', 'tickers', 'actions', 'sp500']
    
    for table in sharadar_tables:
        try:
            cols = con.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = '{table}'
            """).df()
            if len(cols) > 0:
                row_count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                print(f"\n✓ {table}: {row_count:,} rows")
                print(f"  Columns: {cols['column_name'].tolist()[:15]}{'...' if len(cols) > 15 else ''}")
        except:
            print(f"✗ {table}: not found")

    # =========================================================================
    # 7. WHAT'S MISSING FOR TRUE ALPHA
    # =========================================================================
    print("\n" + "=" * 80)
    print("7. ALPHA FACTOR OPPORTUNITY ASSESSMENT")
    print("=" * 80)
    
    print("""
ACADEMICALLY-PROVEN FACTORS & YOUR DATA STATUS:

Factor                  | Expected IC | Data Source      | Status
------------------------|-------------|------------------|--------
Earnings Surprise (SUE) | 2-4%        | SF1 + estimates  | ?
Accruals Quality        | 3-5%        | SF1 (BS + CF)    | ?
Profitability (ROE/ROA) | 1-2%        | SF1              | ?
Gross Margin Change     | 1-2%        | SF1              | ?
Asset Growth            | 1-2% (neg)  | SF1              | ?
Book/Market Value       | 1-2%        | SF1 + price      | ?
Earnings Yield          | 1-2%        | SF1 + price      | ?
12-1 Month Momentum     | 2-4%        | SEP              | HAVE
Short Interest          | 2-4%        | Need external    | MISSING
Insider Transactions    | 1-3%        | SF2 or external  | ?
Institutional Flow      | 1-2%        | SF3              | HAVE?
Analyst Revisions       | 2-3%        | Need external    | MISSING
Options IV Skew         | 1-2%        | Polygon          | AVAILABLE

HIGHEST PRIORITY TO BUILD:
1. Earnings Quality (Accruals) - Highest IC, you have the data
2. SUE (Earnings Surprise) - High IC if you have estimates
3. Quality Composite (ROE + margins + low debt)
4. Proper 12-1 momentum (skip most recent month)

NEED EXTERNAL DATA:
1. Short Interest - Polygon or other
2. Analyst Estimates - Sharadar or IBES
3. News Sentiment - Polygon
""")

    # =========================================================================
    # 8. QUICK IC TEST OF BASIC FACTORS
    # =========================================================================
    print("\n" + "=" * 80)
    print("8. QUICK IC TEST OF AVAILABLE SIGNALS")
    print("=" * 80)
    
    # Test various columns in feat_matrix
    test_cols = [
        'ret_5d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',  # momentum
        'vol_21', 'vol_63',  # volatility
        'adv_20', 'adv_60',  # liquidity
        'beta_252d',  # beta
    ]
    
    print("\nIC of basic factors vs ret_5d_f:")
    print("-" * 50)
    
    for col in test_cols:
        try:
            ic = con.execute(f"""
                SELECT CORR({col}, ret_5d_f) as ic
                FROM feat_matrix
                WHERE {col} IS NOT NULL AND ret_5d_f IS NOT NULL
                AND date >= '2015-01-01'
            """).fetchone()[0]
            if ic is not None:
                strength = "STRONG" if abs(ic) > 0.02 else "MODERATE" if abs(ic) > 0.01 else "WEAK"
                print(f"  {col:<15}: IC = {ic:>8.4f}  ({strength})")
        except Exception as e:
            print(f"  {col:<15}: error - {e}")

    con.close()
    
    print("\n" + "=" * 80)
    print("AUDIT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()