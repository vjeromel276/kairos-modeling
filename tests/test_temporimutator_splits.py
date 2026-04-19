"""Unit tests for scripts/temporimutator/splits.py — leakage audit is the gate."""

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.temporimutator.splits import (
    TRAIN_CUTOFF,
    VAL_END,
    VAL_START,
    TEST_END,
    TEST_START,
    assign_split,
    audit_leakage,
)


def _synthetic_meta():
    """Build meta with train/val/test windows that respect the purge gap.

    Uses short compressed offsets (not realistic 252 trading days) so that
    train labels comfortably clear VAL_START. We're testing the audit logic,
    not simulating real data density.
    """
    rows = []
    # Train: window_start < TRAIN_CUTOFF; labels clear VAL_START by months
    for d in pd.date_range("2018-01-02", TRAIN_CUTOFF - pd.Timedelta(days=1), freq="21D"):
        rows.append({
            "split": "train",
            "window_start": d,
            "window_end": d + pd.Timedelta(days=60),
            "label_date_5": d + pd.Timedelta(days=65),
            "label_date_10": d + pd.Timedelta(days=70),
            "label_date_20": d + pd.Timedelta(days=80),
        })
    # Val
    for d in pd.date_range(VAL_START, VAL_END, freq="21D"):
        rows.append({
            "split": "val",
            "window_start": d,
            "window_end": d + pd.Timedelta(days=60),
            "label_date_5": d + pd.Timedelta(days=65),
            "label_date_10": d + pd.Timedelta(days=70),
            "label_date_20": d + pd.Timedelta(days=80),
        })
    # Test
    for d in pd.date_range(TEST_START, TEST_END, freq="21D"):
        rows.append({
            "split": "test",
            "window_start": d,
            "window_end": d + pd.Timedelta(days=60),
            "label_date_5": d + pd.Timedelta(days=65),
            "label_date_10": d + pd.Timedelta(days=70),
            "label_date_20": d + pd.Timedelta(days=80),
        })
    return pd.DataFrame(rows)


def test_assign_split_boundaries():
    assert assign_split(TRAIN_CUTOFF - pd.Timedelta(days=1)) == "train"
    assert assign_split(TRAIN_CUTOFF) == "purge"       # purge gap between train and val
    assert assign_split(VAL_START) == "val"
    assert assign_split(VAL_END) == "val"
    # Val → test is contiguous per spec (no hard purge, only soft warning)
    assert assign_split(VAL_END + pd.Timedelta(days=1)) == "test"
    assert assign_split(TEST_START) == "test"
    assert assign_split(TEST_END) == "test"
    assert assign_split(TEST_END + pd.Timedelta(days=1)) == "purge"


def test_audit_leakage_passes_on_canonical_split():
    meta = _synthetic_meta()
    audit_leakage(meta)  # should not raise


def test_audit_leakage_fails_when_train_spills_into_val():
    meta = _synthetic_meta()
    # Force one train row to have a label_date past VAL_START
    contaminator = meta[meta["split"] == "train"].iloc[-1].name
    meta.loc[contaminator, "label_date_20"] = VAL_START + pd.Timedelta(days=10)
    with pytest.raises(AssertionError, match="train label_date max"):
        audit_leakage(meta)


def test_audit_leakage_warns_on_val_test_overlap(caplog):
    import logging
    meta = _synthetic_meta()
    contaminator = meta[meta["split"] == "val"].iloc[-1].name
    meta.loc[contaminator, "label_date_20"] = TEST_START + pd.Timedelta(days=10)
    with caplog.at_level(logging.WARNING):
        audit_leakage(meta)
    assert any("val→test feature overlap" in rec.message for rec in caplog.records)
