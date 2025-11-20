#!/usr/bin/env python3
import sys

def test_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError:
        print(f"✗ {module_name}")
        return False

print("Testing pipeline dependencies:\n")
test_import("pandas")
test_import("numpy") 
test_import("duckdb")
test_import("polars")
test_import("requests")
test_import("sklearn")
test_import("lightgbm")
test_import("xgboost")
test_import("torch")
test_import("statsmodels")
test_import("sktime")
test_import("tslearn")
test_import("wandb")
test_import("optuna")
test_import("tqdm")
test_import("yfinance")
