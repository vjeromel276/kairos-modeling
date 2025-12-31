#!/usr/bin/env python3
"""
validate_pipeline_data.py
=========================
Comprehensive validation of the entire Kairos data pipeline.

Checks:
1. Source tables (sep, sf1, daily, etc.) - structure and date types
2. All feat_* tables - date types, coverage, row counts
3. Composite tables - dependencies and join compatibility
4. Final matrices - coverage of key alpha columns
5. Identifies DATE vs TIMESTAMP_NS mismatches that cause silent join failures

Usage:
    python scripts/validate_pipeline_data.py --db data/kairos.duckdb
    python scripts/validate_pipeline_data.py --db data/kairos.duckdb --fix  # Auto-fix date types
"""

import argparse
import duckdb
import sys
from datetime import datetime


def print_header(title):
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")


def print_subheader(title):
    print(f"\n{'-'*60}")
    print(f" {title}")
    print(f"{'-'*60}")


def check_table_exists(con, table_name):
    """Check if a table exists."""
    result = con.execute(f"""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_schema = 'main' AND table_name = '{table_name}'
    """).fetchone()[0]
    return result > 0


def get_date_type(con, table_name):
    """Get the date column type for a table."""
    try:
        result = con.execute(f"SELECT typeof(date) FROM {table_name} LIMIT 1").fetchone()
        return result[0] if result else None
    except:
        return None


def get_table_stats(con, table_name):
    """Get basic stats for a table."""
    try:
        stats = con.execute(f"""
            SELECT 
                COUNT(*) as row_count,
                COUNT(DISTINCT ticker) as ticker_count,
                MIN(date) as min_date,
                MAX(date) as max_date
            FROM {table_name}
        """).fetchone()
        return {
            'row_count': stats[0],
            'ticker_count': stats[1],
            'min_date': stats[2],
            'max_date': stats[3]
        }
    except Exception as e:
        return {'error': str(e)}


def fix_date_type(con, table_name):
    """Fix TIMESTAMP_NS to DATE for a table."""
    try:
        con.execute(f"ALTER TABLE {table_name} ALTER COLUMN date TYPE DATE")
        return True
    except Exception as e:
        print(f"    ⚠ Could not ALTER, trying recreate: {e}")
        try:
            # Recreate table with correct type
            con.execute(f"CREATE TABLE {table_name}_backup AS SELECT * FROM {table_name}")
            con.execute(f"DROP TABLE {table_name}")
            con.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT * REPLACE (CAST(date AS DATE) AS date) 
                FROM {table_name}_backup
            """)
            con.execute(f"DROP TABLE {table_name}_backup")
            return True
        except Exception as e2:
            print(f"    ✗ Failed to fix: {e2}")
            return False


def validate_source_tables(con, fix=False):
    """Validate source data tables."""
    print_header("1. SOURCE TABLES")
    
    source_tables = {
        'sep': 'Primary price data',
        'sep_base': 'Base price data (academic)',
        'sep_base_academic': 'Academic universe base',
        'sf1': 'Fundamental data (quarterly)',
        'sf2': 'Institutional holdings',
        'daily': 'Daily market data',
    }
    
    issues = []
    
    for table, description in source_tables.items():
        if not check_table_exists(con, table):
            print(f"  ✗ {table}: NOT FOUND - {description}")
            issues.append((table, 'missing'))
            continue
        
        date_type = get_date_type(con, table)
        stats = get_table_stats(con, table)
        
        if 'error' in stats:
            print(f"  ⚠ {table}: ERROR - {stats['error']}")
            issues.append((table, 'error'))
            continue
        
        type_status = "✓" if date_type == "DATE" else "✗ TIMESTAMP_NS"
        print(f"  {type_status} {table}: {stats['row_count']:,} rows, {stats['ticker_count']:,} tickers, {stats['min_date']} to {stats['max_date']}")
        
        if date_type != "DATE" and date_type is not None:
            issues.append((table, 'date_type'))
            if fix:
                print(f"    → Fixing date type...")
                if fix_date_type(con, table):
                    print(f"    ✓ Fixed")
                    issues.remove((table, 'date_type'))
    
    return issues


def validate_feature_tables(con, fix=False):
    """Validate all feat_* tables."""
    print_header("2. FEATURE TABLES")
    
    # Get all feat_ tables
    tables = con.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'main' AND table_name LIKE 'feat_%'
        ORDER BY table_name
    """).fetchdf()['table_name'].tolist()
    
    print(f"\n  Found {len(tables)} feature tables\n")
    
    issues = []
    results = []
    
    for table in tables:
        date_type = get_date_type(con, table)
        stats = get_table_stats(con, table)
        
        if 'error' in stats:
            print(f"  ⚠ {table}: ERROR - {stats['error']}")
            issues.append((table, 'error'))
            continue
        
        is_date = date_type == "DATE"
        type_icon = "✓" if is_date else "✗"
        
        results.append({
            'table': table,
            'date_type': date_type,
            'is_date': is_date,
            'row_count': stats['row_count'],
            'ticker_count': stats['ticker_count'],
            'min_date': stats['min_date'],
            'max_date': stats['max_date']
        })
        
        if not is_date and date_type is not None:
            issues.append((table, 'date_type'))
    
    # Print summary grouped by date type status
    print_subheader("Tables with correct DATE type")
    for r in sorted(results, key=lambda x: x['table']):
        if r['is_date']:
            print(f"  ✓ {r['table']}: {r['row_count']:,} rows")
    
    print_subheader("Tables with TIMESTAMP_NS (NEEDS FIX)")
    timestamp_tables = [r for r in results if not r['is_date']]
    if timestamp_tables:
        for r in sorted(timestamp_tables, key=lambda x: x['table']):
            print(f"  ✗ {r['table']}: {r['row_count']:,} rows - {r['date_type']}")
            if fix:
                print(f"    → Fixing date type...")
                if fix_date_type(con, r['table']):
                    print(f"    ✓ Fixed")
                    issues = [(t, i) for t, i in issues if t != r['table']]
    else:
        print("  None - all tables have correct DATE type")
    
    return issues


def validate_regime_tables(con, fix=False):
    """Validate regime-related tables."""
    print_header("3. REGIME TABLES")
    
    regime_tables = [
        'regime_history_academic',
        'feat_vol_sizing',
    ]
    
    issues = []
    
    for table in regime_tables:
        if not check_table_exists(con, table):
            print(f"  ⚠ {table}: NOT FOUND")
            continue
        
        date_type = get_date_type(con, table)
        
        # Get regime distribution
        try:
            regimes = con.execute(f"""
                SELECT regime, COUNT(*) as cnt 
                FROM {table} 
                GROUP BY regime 
                ORDER BY cnt DESC
            """).fetchdf()
            
            type_status = "✓" if date_type == "DATE" else "✗ TIMESTAMP_NS"
            print(f"\n  {type_status} {table}:")
            print(f"    Regimes: {regimes['regime'].tolist()}")
            print(f"    Date type: {date_type}")
            
            if date_type != "DATE":
                issues.append((table, 'date_type'))
                if fix:
                    print(f"    → Fixing date type...")
                    if fix_date_type(con, table):
                        print(f"    ✓ Fixed")
                        issues.remove((table, 'date_type'))
        except Exception as e:
            print(f"  ⚠ {table}: ERROR - {e}")
    
    return issues


def validate_composite_dependencies(con):
    """Check that composite table dependencies have matching date types."""
    print_header("4. COMPOSITE DEPENDENCIES")
    
    # Define dependency chains
    dependencies = {
        'feat_composite_v33_regime': ['feat_composite_academic', 'feat_composite_long_v2', 'feat_alpha_smoothed_v31', 'regime_history_academic'],
        'feat_composite_v7': ['feat_composite_v33_regime', 'feat_quality_v2'],
        'feat_composite_v8': ['feat_composite_v33_regime', 'feat_quality_v2', 'feat_value_v2'],
    }
    
    issues = []
    
    for composite, deps in dependencies.items():
        print(f"\n  {composite}:")
        
        if not check_table_exists(con, composite):
            print(f"    ⚠ NOT FOUND")
            continue
        
        composite_type = get_date_type(con, composite)
        print(f"    Date type: {composite_type}")
        
        all_match = True
        for dep in deps:
            if not check_table_exists(con, dep):
                print(f"    ⚠ {dep}: NOT FOUND")
                all_match = False
                continue
            
            dep_type = get_date_type(con, dep)
            match = "✓" if dep_type == composite_type else "✗ MISMATCH"
            print(f"    {match} {dep}: {dep_type}")
            
            if dep_type != composite_type:
                all_match = False
                issues.append((composite, dep, 'type_mismatch'))
        
        if all_match:
            print(f"    ✓ All dependencies have matching date types")
    
    return issues


def validate_matrix_coverage(con):
    """Check coverage of key columns in feat_matrix_v2."""
    print_header("5. FEATURE MATRIX COVERAGE")
    
    if not check_table_exists(con, 'feat_matrix_v2'):
        print("  ✗ feat_matrix_v2 NOT FOUND")
        return []
    
    # Key alpha columns to check
    alpha_columns = [
        'alpha_composite_v33_regime',
        'alpha_composite_v8',
        'alpha_composite_v7',
        'alpha_composite_eq',
        'alpha_CL_v2',
        'alpha_smoothed',
        'quality_composite_z',
        'value_composite_z',
        'ret_5d_f',
        'adv_20',
    ]
    
    # Get available columns
    available = con.execute("""
        SELECT column_name FROM information_schema.columns 
        WHERE table_name = 'feat_matrix_v2'
    """).fetchdf()['column_name'].tolist()
    
    print(f"\n  Checking coverage for key columns (date >= 2015-01-01):\n")
    print(f"  {'Column':<35} {'Coverage':>10} {'Status':>10}")
    print(f"  {'-'*55}")
    
    issues = []
    
    for col in alpha_columns:
        if col not in available:
            print(f"  {col:<35} {'N/A':>10} {'MISSING':>10}")
            issues.append((col, 'missing'))
            continue
        
        try:
            result = con.execute(f"""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN "{col}" IS NOT NULL THEN 1 ELSE 0 END) as non_null
                FROM feat_matrix_v2
                WHERE date >= '2015-01-01'
            """).fetchone()
            
            total, non_null = result
            pct = 100.0 * non_null / total if total > 0 else 0
            
            if pct >= 95:
                status = "✓ OK"
            elif pct >= 80:
                status = "⚠ LOW"
                issues.append((col, 'low_coverage'))
            else:
                status = "✗ CRITICAL"
                issues.append((col, 'critical_coverage'))
            
            print(f"  {col:<35} {pct:>9.1f}% {status:>10}")
        except Exception as e:
            print(f"  {col:<35} {'ERROR':>10} {str(e)[:20]:>10}")
            issues.append((col, 'error'))
    
    return issues


def validate_date_consistency_across_joins(con):
    """Test actual join results between related tables."""
    print_header("6. JOIN COMPATIBILITY TEST")
    
    # Test joins that are known to be problematic
    join_tests = [
        {
            'name': 'v33 components',
            'query': """
                SELECT COUNT(*) as matched
                FROM feat_composite_academic a
                JOIN feat_composite_long_v2 b ON a.ticker = b.ticker AND a.date = b.date
                WHERE a.date >= '2017-01-01' AND a.date <= '2017-12-31'
            """
        },
        {
            'name': 'v33 + regime',
            'query': """
                SELECT COUNT(*) as matched
                FROM feat_composite_academic a
                JOIN regime_history_academic r ON a.date = r.date
                WHERE a.date >= '2017-01-01' AND a.date <= '2017-12-31'
            """
        },
        {
            'name': 'v7 components (v33 + quality)',
            'query': """
                SELECT COUNT(*) as matched
                FROM feat_composite_v33_regime a
                JOIN feat_quality_v2 b ON a.ticker = b.ticker AND a.date = b.date
                WHERE a.date >= '2017-01-01' AND a.date <= '2017-12-31'
            """
        },
        {
            'name': 'matrix + v33',
            'query': """
                SELECT COUNT(*) as matched
                FROM feat_matrix_v2 m
                JOIN feat_composite_v33_regime v ON m.ticker = v.ticker AND m.date = v.date
                WHERE m.date >= '2017-01-01' AND m.date <= '2017-12-31'
            """
        },
    ]
    
    issues = []
    
    for test in join_tests:
        try:
            result = con.execute(test['query']).fetchone()[0]
            status = "✓" if result > 0 else "✗ NO MATCHES"
            print(f"  {status} {test['name']}: {result:,} rows matched")
            
            if result == 0:
                issues.append((test['name'], 'no_matches'))
        except Exception as e:
            print(f"  ⚠ {test['name']}: ERROR - {str(e)[:50]}")
            issues.append((test['name'], 'error'))
    
    return issues


def generate_fix_script(all_issues):
    """Generate SQL commands to fix identified issues."""
    print_header("7. RECOMMENDED FIXES")
    
    date_type_issues = [(t, i) for t, i in all_issues if i == 'date_type']
    
    if not date_type_issues:
        print("\n  ✓ No date type issues to fix")
        return
    
    print("\n  Run these SQL commands to fix date type issues:\n")
    
    for table, _ in date_type_issues:
        print(f"  ALTER TABLE {table} ALTER COLUMN date TYPE DATE;")
    
    print("\n  Or run this script with --fix flag to auto-fix.")


def main():
    parser = argparse.ArgumentParser(description="Validate Kairos pipeline data")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--fix", action="store_true", help="Auto-fix date type issues")
    args = parser.parse_args()

    print(f"\n{'#'*80}")
    print(f"# KAIROS PIPELINE DATA VALIDATION")
    print(f"# Database: {args.db}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Auto-fix: {'ENABLED' if args.fix else 'disabled'}")
    print(f"{'#'*80}")

    con = duckdb.connect(args.db)
    
    all_issues = []
    
    # Run all validations
    all_issues.extend(validate_source_tables(con, fix=args.fix))
    all_issues.extend(validate_feature_tables(con, fix=args.fix))
    all_issues.extend(validate_regime_tables(con, fix=args.fix))
    all_issues.extend(validate_composite_dependencies(con))
    all_issues.extend(validate_matrix_coverage(con))
    all_issues.extend(validate_date_consistency_across_joins(con))
    
    # Summary
    print_header("SUMMARY")
    
    if not all_issues:
        print("\n  ✓ All validations passed - pipeline data is healthy")
    else:
        print(f"\n  Found {len(all_issues)} issue(s):\n")
        
        # Group by issue type
        by_type = {}
        for item in all_issues:
            if len(item) == 2:
                table, issue = item
                key = issue
            else:
                table, dep, issue = item
                key = issue
            
            if key not in by_type:
                by_type[key] = []
            by_type[key].append(item)
        
        for issue_type, items in by_type.items():
            print(f"  {issue_type}: {len(items)} table(s)")
            for item in items[:5]:  # Show first 5
                if len(item) == 2:
                    print(f"    - {item[0]}")
                else:
                    print(f"    - {item[0]} <-> {item[1]}")
            if len(items) > 5:
                print(f"    ... and {len(items) - 5} more")
        
        if not args.fix:
            generate_fix_script(all_issues)
    
    con.close()
    
    print(f"\n{'#'*80}")
    print(f"# VALIDATION COMPLETE")
    print(f"{'#'*80}\n")
    
    return 0 if not all_issues else 1


if __name__ == "__main__":
    sys.exit(main())