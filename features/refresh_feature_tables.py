#!/usr/bin/env python3
"""
features/refresh_feature_tables.py

Runs all available feature engineering scripts to refresh feat_* tables in DuckDB.
Scalable: just add more scripts to the list below.
"""

import subprocess
import sys

FEATURE_SCRIPTS = [
    "features/price_action_features.py",
    "features/price_shape_features.py",
    "features/statistical_features.py",
    "features/trend_features.py",
    "features/volume_volatility_features.py",
    "features/fundamental_features.py",
    "features/ownership_features.py",
    "features/quality_features.py",
    # "features/analyst_features.py",
    # "features/macro_features.py",
    # "features/polygon_intraday_features.py"

]

DB_PATH = "data/kairos.duckdb"

print("🚀 Refreshing all feature tables...")

for script in FEATURE_SCRIPTS:
    print(f"\n▶ Running {script}")
    result = subprocess.run([sys.executable, script, "--db", DB_PATH])
    if result.returncode != 0:
        print(f"❌ {script} failed. Halting.")
        sys.exit(result.returncode)

print("\n✅ All feature tables refreshed successfully.")
