#!/usr/bin/env python3
"""
Kairos Rebalance Schedule Checker v2

Rebalance on the LAST TRADING DAY of each week.
- Usually Friday
- Thursday if Friday is a holiday
- Wednesday if Thu+Fri are holidays (e.g., Thanksgiving week)

Usage:
    python check_rebalance.py --date 2025-12-30
    python check_rebalance.py --next 20
    python check_rebalance.py --range 2026-01-01 2026-06-30

Author: Kairos Quant Engineering
"""

import argparse
from datetime import datetime, timedelta
import pandas as pd

# Try to use pandas_market_calendars for accurate NYSE holidays
try:
    import pandas_market_calendars as mcal
    HAS_MARKET_CAL = True
    NYSE = mcal.get_calendar('NYSE')
except ImportError:
    HAS_MARKET_CAL = False
    NYSE = None

# Known NYSE holidays for 2024-2026 (fallback if no market calendar)
NYSE_HOLIDAYS = {
    # 2024
    '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29', 
    '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
    '2024-11-28', '2024-12-25',
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
    '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
    '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03',
    '2026-05-25', '2026-06-19', '2026-07-03', '2026-09-07',
    '2026-11-26', '2026-12-25',
}


def is_trading_day(date):
    """Check if a date is a trading day (not weekend, not holiday)."""
    d = pd.Timestamp(date)
    
    # Weekend check
    if d.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Holiday check
    if HAS_MARKET_CAL:
        # Use market calendar
        schedule = NYSE.schedule(start_date=d, end_date=d)
        return len(schedule) > 0
    else:
        # Use hardcoded holidays
        return d.strftime('%Y-%m-%d') not in NYSE_HOLIDAYS


def get_trading_days_in_week(date):
    """Get all trading days in the same week as the given date."""
    d = pd.Timestamp(date)
    
    # Find Monday of this week
    monday = d - timedelta(days=d.weekday())
    
    # Get Mon-Fri of this week
    week_days = [monday + timedelta(days=i) for i in range(5)]
    
    # Filter to trading days only
    trading_days = [day for day in week_days if is_trading_day(day)]
    
    return trading_days


def get_last_trading_day_of_week(date):
    """Get the last trading day of the week containing the given date."""
    trading_days = get_trading_days_in_week(date)
    if trading_days:
        return trading_days[-1]
    return None


def is_rebalance_day(date):
    """Check if date is a rebalance day (last trading day of week)."""
    d = pd.Timestamp(date)
    
    if not is_trading_day(d):
        return False
    
    last_day = get_last_trading_day_of_week(d)
    return last_day is not None and d.date() == last_day.date()


def get_next_rebalance_dates(from_date, n=10):
    """Get next N rebalance dates (last trading day of each week)."""
    current = pd.Timestamp(from_date)
    rebalance_dates = []
    
    # Move to start of current week
    monday = current - timedelta(days=current.weekday())
    
    weeks_checked = 0
    while len(rebalance_dates) < n and weeks_checked < 100:
        week_start = monday + timedelta(weeks=weeks_checked)
        last_day = get_last_trading_day_of_week(week_start)
        
        if last_day is not None and last_day >= current:
            rebalance_dates.append(last_day)
        
        weeks_checked += 1
    
    return rebalance_dates


def get_rebalance_dates_in_range(start_date, end_date):
    """Get all rebalance dates in a date range."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    
    # Start from Monday of start week
    monday = start - timedelta(days=start.weekday())
    
    rebalance_dates = []
    current_monday = monday
    
    while current_monday <= end:
        last_day = get_last_trading_day_of_week(current_monday)
        if last_day is not None and start <= last_day <= end:
            rebalance_dates.append(last_day)
        current_monday += timedelta(weeks=1)
    
    return rebalance_dates


def main():
    parser = argparse.ArgumentParser(description='Check Kairos rebalance schedule')
    parser.add_argument('--date', help='Check if specific date is rebalance day')
    parser.add_argument('--next', type=int, default=10, help='Show next N rebalance dates')
    parser.add_argument('--range', nargs=2, metavar=('START', 'END'), 
                       help='Show rebalance dates in range')
    parser.add_argument('--from-date', help='Starting date for --next (default: today)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("KAIROS REBALANCE SCHEDULE")
    print("=" * 60)
    print("Rule: Last trading day of each week")
    print(f"Calendar: {'NYSE (pandas_market_calendars)' if HAS_MARKET_CAL else 'NYSE (hardcoded holidays)'}")
    print("-" * 60)
    
    if args.date:
        date = pd.Timestamp(args.date)
        is_rebal = is_rebalance_day(date)
        is_trading = is_trading_day(date)
        last_of_week = get_last_trading_day_of_week(date)
        
        print(f"\nDate: {date.date()} ({date.strftime('%A')})")
        print(f"Is trading day: {'Yes' if is_trading else 'No (weekend/holiday)'}")
        print(f"Last trading day of week: {last_of_week.date() if last_of_week else 'N/A'} ({last_of_week.strftime('%A') if last_of_week else ''})")
        print(f"\nIS REBALANCE DAY: {'✓ YES' if is_rebal else '✗ NO'}")
        
        if not is_rebal:
            # Find next rebalance
            next_dates = get_next_rebalance_dates(date, 1)
            if next_dates:
                next_date = next_dates[0]
                print(f"Next rebalance: {next_date.date()} ({next_date.strftime('%A')})")
    
    elif args.range:
        start, end = args.range
        dates = get_rebalance_dates_in_range(start, end)
        print(f"\nRebalance dates from {start} to {end}:")
        print(f"Total: {len(dates)} rebalances\n")
        for i, d in enumerate(dates, 1):
            weekday = d.strftime('%A')
            print(f"  {i:3d}. {d.date()} ({weekday})")
    
    else:
        from_date = args.from_date or datetime.now().strftime('%Y-%m-%d')
        dates = get_next_rebalance_dates(from_date, args.next)
        
        print(f"\nNext {args.next} rebalance dates from {from_date}:\n")
        for i, d in enumerate(dates, 1):
            weekday = d.strftime('%A')
            print(f"  {i:3d}. {d.date()} ({weekday})")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
