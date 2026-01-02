#!/usr/bin/env python3
"""
rebuild_feat_fundamental.py
============================
Rebuild feat_fundamental table with proper forward-fill from SF1 quarterly data.

Problem: Current feat_fundamental has 1.6% coverage because quarterly data
         is not forward-filled to daily trading dates.

Solution: 
1. Load SF1 (ARQ dimension) quarterly fundamentals
2. Compute derived ratios (ROE, ROA, etc. - these are 0% in raw SF1)
3. Compute YoY growth rates
4. Forward-fill to daily dates in sep_base_academic
5. Result: ~95%+ coverage on daily trading dates

Usage:
    python scripts/features/rebuild_feat_fundamental.py --db data/kairos.duckdb
    python scripts/features/rebuild_feat_fundamental.py --db data/kairos.duckdb --dry-run
"""

import argparse
import logging
import duckdb
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_quarterly_fundamentals(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Extract and compute fundamental ratios from SF1 quarterly data.
    """
    logger.info("Extracting SF1 quarterly data (ARQ dimension)...")
    
    # Get raw quarterly data
    # Using datekey as the "known from" date (when data becomes available)
    df = con.execute("""
        SELECT 
            ticker,
            datekey as date,
            calendardate,
            
            -- Valuation inputs (for computing yields)
            pe,
            pb,
            ps,
            ev,
            evebit,
            evebitda,
            marketcap,
            
            -- Profitability inputs
            revenue,
            netinc,
            gp,
            ebit,
            ebitda,
            
            -- Balance sheet
            assets,
            equity,
            debt,
            
            -- Cash flow
            ncfo,
            fcf,
            capex,
            
            -- Pre-computed margins (these exist in SF1)
            grossmargin,
            netmargin,
            ebitdamargin,
            
            -- Other ratios
            de,
            currentratio
            
        FROM sf1
        WHERE dimension = 'ARQ'
          AND datekey IS NOT NULL
        ORDER BY ticker, datekey
    """).fetchdf()
    
    logger.info(f"Loaded {len(df):,} quarterly records")
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    df['calendardate'] = pd.to_datetime(df['calendardate'])
    
    # =========================================================================
    # COMPUTE DERIVED RATIOS (these are 0% in SF1)
    # =========================================================================
    logger.info("Computing derived ratios...")
    
    # Profitability ratios
    df['roe'] = df['netinc'] / df['equity'].replace(0, np.nan)
    df['roa'] = df['netinc'] / df['assets'].replace(0, np.nan)
    df['asset_turnover'] = df['revenue'] / df['assets'].replace(0, np.nan)
    
    # Use pre-computed margins from SF1, but rename for consistency
    df['gross_margin'] = df['grossmargin']
    df['operating_margin'] = df['ebit'] / df['revenue'].replace(0, np.nan)
    df['net_margin'] = df['netmargin']
    
    # Valuation yields (inverse of ratios)
    df['earnings_yield'] = 1.0 / df['pe'].replace(0, np.nan)
    df['book_to_market'] = 1.0 / df['pb'].replace(0, np.nan)
    df['sales_to_price'] = 1.0 / df['ps'].replace(0, np.nan)
    df['fcf_yield'] = df['fcf'] / df['marketcap'].replace(0, np.nan)
    df['ebitda_to_ev'] = df['ebitda'] / df['ev'].replace(0, np.nan)
    
    # Quality
    df['debt_to_equity'] = df['de']  # Already in SF1
    df['accruals'] = (df['netinc'] - df['ncfo']) / df['assets'].replace(0, np.nan)
    
    # =========================================================================
    # COMPUTE YoY GROWTH RATES
    # =========================================================================
    logger.info("Computing YoY growth rates...")
    
    # Sort for proper lag calculation
    df = df.sort_values(['ticker', 'date'])
    
    # Lag by 4 quarters for YoY
    for col, new_col in [
        ('revenue', 'revenue_growth_yoy'),
        ('netinc', 'earnings_growth_yoy'),
        ('fcf', 'fcf_growth_yoy'),
        ('equity', 'book_value_growth_yoy'),
    ]:
        df[f'{col}_lag4'] = df.groupby('ticker')[col].shift(4)
        df[new_col] = (df[col] - df[f'{col}_lag4']) / df[f'{col}_lag4'].abs().replace(0, np.nan)
        df.drop(columns=[f'{col}_lag4'], inplace=True)
    
    # =========================================================================
    # COMPUTE TRAILING VOLATILITY (4-quarter rolling std)
    # =========================================================================
    logger.info("Computing volatility metrics...")
    
    df['earnings_volatility'] = df.groupby('ticker')['netinc'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )
    df['revenue_volatility'] = df.groupby('ticker')['revenue'].transform(
        lambda x: x.rolling(4, min_periods=2).std()
    )
    
    # =========================================================================
    # SELECT FINAL COLUMNS
    # =========================================================================
    output_cols = [
        'ticker', 'date',
        # Valuation
        'earnings_yield', 'fcf_yield', 'book_to_market', 'sales_to_price', 'ebitda_to_ev',
        # Profitability
        'roe', 'roa', 'gross_margin', 'operating_margin', 'net_margin',
        # Efficiency & Leverage
        'asset_turnover', 'debt_to_equity', 'accruals',
        # Growth
        'revenue_growth_yoy', 'earnings_growth_yoy', 'fcf_growth_yoy', 'book_value_growth_yoy',
        # Volatility
        'earnings_volatility', 'revenue_volatility',
    ]
    
    result = df[output_cols].copy()
    
    # Clip extreme values to prevent outliers from dominating
    numeric_cols = [c for c in output_cols if c not in ['ticker', 'date']]
    for col in numeric_cols:
        if col in result.columns:
            # Clip at 1st and 99th percentile
            p01 = result[col].quantile(0.01)
            p99 = result[col].quantile(0.99)
            result[col] = result[col].clip(p01, p99)
    
    logger.info(f"Quarterly fundamentals computed: {len(result):,} rows")
    
    return result


def forward_fill_to_daily(
    con: duckdb.DuckDBPyConnection, 
    quarterly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Forward-fill quarterly fundamentals to daily trading dates.
    
    For each ticker:
    - Get all trading dates from sep_base_academic
    - For each trading date, use the most recent quarterly data (where date <= trading_date)
    """
    logger.info("Forward-filling to daily trading dates...")
    
    # Get universe of (ticker, date) from sep_base_academic
    daily_grid = con.execute("""
        SELECT DISTINCT ticker, date
        FROM sep_base_academic
        WHERE date >= '2000-01-01'
        ORDER BY ticker, date
    """).fetchdf()
    
    daily_grid['date'] = pd.to_datetime(daily_grid['date'])
    
    logger.info(f"Daily grid: {len(daily_grid):,} (ticker, date) pairs")
    
    # Merge with as-of join (forward fill)
    # For each daily date, find the most recent quarterly date <= daily date
    
    quarterly_df = quarterly_df.sort_values(['ticker', 'date']).reset_index(drop=True)
    daily_grid = daily_grid.sort_values(['ticker', 'date']).reset_index(drop=True)
    
    # merge_asof requires sorted keys - sort by the 'by' column first, then 'on' column
    # Since we have multiple tickers, we need to do this per-ticker or ensure proper sorting
    
    # Process in chunks by ticker for memory efficiency and correct sorting
    logger.info("Processing forward-fill by ticker (this may take a few minutes)...")
    
    tickers = daily_grid['ticker'].unique()
    results = []
    
    for i, ticker in enumerate(tickers):
        if (i + 1) % 500 == 0:
            logger.info(f"  Processed {i+1}/{len(tickers)} tickers...")
        
        daily_ticker = daily_grid[daily_grid['ticker'] == ticker].copy()
        quarterly_ticker = quarterly_df[quarterly_df['ticker'] == ticker].copy()
        
        if len(quarterly_ticker) == 0:
            continue
        
        # merge_asof on single ticker - keys are just dates now
        merged = pd.merge_asof(
            daily_ticker.sort_values('date'),
            quarterly_ticker.sort_values('date'),
            on='date',
            direction='backward',
            tolerance=pd.Timedelta(days=400),
            suffixes=('', '_quarterly')
        )
        
        # Handle duplicate ticker column from merge
        if 'ticker_quarterly' in merged.columns:
            merged = merged.drop(columns=['ticker_quarterly'])
        
        results.append(merged)
    
    result = pd.concat(results, ignore_index=True)
    logger.info(f"  Processed {len(tickers)} tickers")
    
    logger.info(f"After forward-fill: {len(result):,} rows")
    
    # Check coverage
    numeric_cols = [c for c in result.columns if c not in ['ticker', 'date']]
    for col in numeric_cols[:5]:  # Just show first 5
        coverage = result[col].notna().mean() * 100
        logger.info(f"  {col}: {coverage:.1f}% coverage")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Rebuild feat_fundamental with forward-fill")
    parser.add_argument("--db", required=True, help="Path to DuckDB database")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    parser.add_argument("--min-date", default="2000-01-01", help="Minimum date")
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("REBUILD FEAT_FUNDAMENTAL")
    logger.info("=" * 70)
    
    con = duckdb.connect(args.db)
    
    try:
        # Step 1: Build quarterly fundamentals from SF1
        quarterly_df = build_quarterly_fundamentals(con)
        
        # Step 2: Forward-fill to daily
        daily_df = forward_fill_to_daily(con, quarterly_df)
        
        # Step 3: Filter to valid date range
        daily_df = daily_df[daily_df['date'] >= args.min_date]
        logger.info(f"After date filter: {len(daily_df):,} rows")
        
        if args.dry_run:
            logger.info("\n[DRY RUN] Would create table with:")
            logger.info(f"  Rows: {len(daily_df):,}")
            logger.info(f"  Columns: {list(daily_df.columns)}")
            logger.info("\nSample data:")
            print(daily_df.head(10).to_string())
            
            # Show coverage stats
            logger.info("\nCoverage statistics:")
            for col in daily_df.columns:
                if col not in ['ticker', 'date']:
                    cov = daily_df[col].notna().mean() * 100
                    logger.info(f"  {col}: {cov:.1f}%")
        else:
            # Backup old table
            logger.info("\nBacking up old feat_fundamental...")
            con.execute("DROP TABLE IF EXISTS feat_fundamental_backup")
            try:
                con.execute("ALTER TABLE feat_fundamental RENAME TO feat_fundamental_backup")
                logger.info("  Old table backed up to feat_fundamental_backup")
            except:
                logger.info("  No existing table to backup")
            
            # Create new table
            logger.info("Creating new feat_fundamental table...")
            con.register("daily_df", daily_df)
            con.execute("""
                CREATE TABLE feat_fundamental AS
                SELECT * REPLACE (CAST(date AS DATE) AS date)
                FROM daily_df
            """)
            
            # Verify
            count = con.execute("SELECT COUNT(*) FROM feat_fundamental").fetchone()[0]
            logger.info(f"Created feat_fundamental with {count:,} rows")
            
            # Show coverage
            logger.info("\nFinal coverage statistics:")
            for col in daily_df.columns:
                if col not in ['ticker', 'date']:
                    result = con.execute(f"""
                        SELECT 
                            100.0 * SUM(CASE WHEN {col} IS NOT NULL THEN 1 ELSE 0 END) / COUNT(*)
                        FROM feat_fundamental
                        WHERE date >= '2015-01-01'
                    """).fetchone()[0]
                    logger.info(f"  {col}: {result:.1f}%")
            
            # Compare to old coverage
            logger.info("\n" + "=" * 70)
            logger.info("COMPARISON: OLD vs NEW")
            logger.info("=" * 70)
            
            old_coverage = con.execute("""
                SELECT COUNT(*) as matched
                FROM sep_base_academic s
                LEFT JOIN feat_fundamental_backup f ON s.ticker = f.ticker AND s.date = f.date
                WHERE s.date >= '2015-01-01' AND f.date IS NOT NULL
            """).fetchone()[0]
            
            new_coverage = con.execute("""
                SELECT COUNT(*) as matched
                FROM sep_base_academic s
                LEFT JOIN feat_fundamental f ON s.ticker = f.ticker AND s.date = f.date
                WHERE s.date >= '2015-01-01' AND f.date IS NOT NULL
            """).fetchone()[0]
            
            total = con.execute("""
                SELECT COUNT(*) FROM sep_base_academic WHERE date >= '2015-01-01'
            """).fetchone()[0]
            
            logger.info(f"OLD feat_fundamental coverage: {100*old_coverage/total:.1f}% ({old_coverage:,} / {total:,})")
            logger.info(f"NEW feat_fundamental coverage: {100*new_coverage/total:.1f}% ({new_coverage:,} / {total:,})")
            logger.info(f"Improvement: {new_coverage - old_coverage:,} more rows ({100*(new_coverage-old_coverage)/total:.1f}%)")
        
        logger.info("\n" + "=" * 70)
        logger.info("DONE")
        logger.info("=" * 70)
        
    finally:
        con.close()


if __name__ == "__main__":
    main()