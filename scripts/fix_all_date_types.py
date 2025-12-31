#!/usr/bin/env python3
"""
fix_all_date_types.py
=====================
Fix all TIMESTAMP_NS date columns to DATE type across the entire database.

This addresses the systemic issue where pandas DataFrames with datetime columns
get written as TIMESTAMP_NS instead of DATE, causing silent join failures.

Usage:
    python scripts/fix_all_date_types.py --db data/kairos.duckdb
    python scripts/fix_all_date_types.py --db data/kairos.duckdb --dry-run  # Preview only
"""

import argparse
import duckdb
from datetime import datetime


def get_tables_with_date_column(con):
    """Get all tables that have a 'date' column."""
    tables = con.execute("""
        SELECT DISTINCT table_name 
        FROM information_schema.columns 
        WHERE table_schema = 'main' 
          AND column_name = 'date'
        ORDER BY table_name
    """).fetchdf()['table_name'].tolist()
    return tables


def get_date_type(con, table_name):
    """Get the date column type for a table."""
    try:
        result = con.execute(f"SELECT typeof(date) FROM {table_name} LIMIT 1").fetchone()
        return result[0] if result else None
    except Exception as e:
        return f"ERROR: {e}"


def fix_date_type(con, table_name, dry_run=False):
    """Fix TIMESTAMP_NS to DATE for a table."""
    if dry_run:
        return True, "DRY RUN - would fix"
    
    try:
        # Try ALTER first (fastest)
        con.execute(f"ALTER TABLE {table_name} ALTER COLUMN date TYPE DATE")
        return True, "ALTER succeeded"
    except Exception as e:
        # If ALTER fails, recreate the table
        try:
            print(f"    ALTER failed, recreating table...")
            
            # Get row count for progress
            row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"    Table has {row_count:,} rows")
            
            # Create backup
            con.execute(f"CREATE TABLE __{table_name}_backup AS SELECT * FROM {table_name}")
            
            # Drop original
            con.execute(f"DROP TABLE {table_name}")
            
            # Recreate with DATE type
            con.execute(f"""
                CREATE TABLE {table_name} AS 
                SELECT * REPLACE (CAST(date AS DATE) AS date) 
                FROM __{table_name}_backup
            """)
            
            # Drop backup
            con.execute(f"DROP TABLE __{table_name}_backup")
            
            return True, "RECREATE succeeded"
        except Exception as e2:
            return False, f"FAILED: {e2}"


def main():
    parser = argparse.ArgumentParser(description="Fix all date type issues")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    args = parser.parse_args()

    print(f"\n{'#'*70}")
    print(f"# FIX ALL DATE TYPES")
    print(f"# Database: {args.db}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Mode: {'DRY RUN (preview only)' if args.dry_run else 'LIVE (will modify database)'}")
    print(f"{'#'*70}\n")

    con = duckdb.connect(args.db)
    
    # Get all tables with date column
    tables = get_tables_with_date_column(con)
    print(f"Found {len(tables)} tables with 'date' column\n")
    
    # Categorize by date type
    correct = []
    needs_fix = []
    errors = []
    
    for table in tables:
        date_type = get_date_type(con, table)
        if date_type == "DATE":
            correct.append(table)
        elif date_type and date_type.startswith("ERROR"):
            errors.append((table, date_type))
        else:
            needs_fix.append((table, date_type))
    
    # Summary
    print(f"{'='*70}")
    print(f" CURRENT STATE")
    print(f"{'='*70}")
    print(f"\n  ✓ Correct (DATE): {len(correct)} tables")
    print(f"  ✗ Needs fix: {len(needs_fix)} tables")
    print(f"  ⚠ Errors: {len(errors)} tables")
    
    if not needs_fix:
        print(f"\n  All tables have correct DATE type. Nothing to fix.")
        con.close()
        return 0
    
    # Show tables needing fix
    print(f"\n{'='*70}")
    print(f" TABLES TO FIX")
    print(f"{'='*70}\n")
    
    for table, dtype in sorted(needs_fix):
        print(f"  {table}: {dtype}")
    
    # Fix tables
    print(f"\n{'='*70}")
    print(f" {'FIXING' if not args.dry_run else 'WOULD FIX'} TABLES")
    print(f"{'='*70}\n")
    
    fixed = []
    failed = []
    
    for i, (table, dtype) in enumerate(needs_fix):
        print(f"  [{i+1}/{len(needs_fix)}] {table}...")
        success, msg = fix_date_type(con, table, dry_run=args.dry_run)
        
        if success:
            print(f"    ✓ {msg}")
            fixed.append(table)
        else:
            print(f"    ✗ {msg}")
            failed.append((table, msg))
    
    # Final summary
    print(f"\n{'='*70}")
    print(f" RESULTS")
    print(f"{'='*70}")
    print(f"\n  Fixed: {len(fixed)} tables")
    print(f"  Failed: {len(failed)} tables")
    
    if failed:
        print(f"\n  Failed tables:")
        for table, msg in failed:
            print(f"    - {table}: {msg}")
    
    # Verify fixes
    if not args.dry_run and fixed:
        print(f"\n{'='*70}")
        print(f" VERIFICATION")
        print(f"{'='*70}\n")
        
        all_good = True
        for table in fixed[:10]:  # Check first 10
            new_type = get_date_type(con, table)
            status = "✓" if new_type == "DATE" else "✗"
            print(f"  {status} {table}: {new_type}")
            if new_type != "DATE":
                all_good = False
        
        if len(fixed) > 10:
            print(f"  ... and {len(fixed) - 10} more")
        
        if all_good:
            print(f"\n  ✓ All verified tables have DATE type")
    
    con.close()
    
    print(f"\n{'#'*70}")
    print(f"# COMPLETE")
    print(f"{'#'*70}\n")
    
    if not args.dry_run:
        print("NEXT STEPS:")
        print("  1. Rebuild composite tables (v33, v7, v8) - they pull from fixed tables")
        print("  2. Rebuild feat_matrix_v2")
        print("  3. Run validation again to confirm all issues resolved")
        print()
        print("Commands:")
        print("  python scripts/features/build_composite_v33_regime.py --db data/kairos.duckdb")
        print("  python scripts/features/build_alpha_composite_v7.py --db data/kairos.duckdb")
        print("  python scripts/features/build_alpha_composite_v8.py --db data/kairos.duckdb")
        print("  python scripts/build_feature_matrix_v2.py --db data/kairos.duckdb --date 2025-12-26 --universe scripts/sep_dataset/feature_sets/option_b_universe.csv")
        print()
    
    return 0 if not failed else 1


if __name__ == "__main__":
    exit(main())