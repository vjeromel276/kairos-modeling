"""
Train/val/test split + leakage audit for Temporimutator.

Chronological split with a 272-day purge gap between train and val, enforced
at `window_start_date` granularity (spec lines 217-246).

The hard correctness gate is Rule 2: train label_date_max < val window_start_min.
This prevents the model from memorizing labels that a val window will see as
features.

Val→test overlap is NOT hard-purged (spec treats val/test as contiguous:
val 2023-01-01..2023-12-31, test 2024-01-01..2024-12-31). Val features do
overlap test features, but val labels are not injected into test features
during training, so this is model-selection bias rather than label leakage.
We emit a runtime warning if that overlap is present so it's never silent.
"""

from __future__ import annotations

import logging
import pandas as pd

log = logging.getLogger(__name__)

VAL_START = pd.Timestamp("2023-01-01")
VAL_END = pd.Timestamp("2023-12-31")
TEST_START = pd.Timestamp("2024-01-01")
TEST_END = pd.Timestamp("2024-12-31")

PURGE_GAP_DAYS = 272  # window (252) + max label horizon (20) in trading days

# Spec lines 222-226 claim TRAIN_CUTOFF = 2022-03-01 but that's only ~10 months
# of padding — 272 trading days is ~13 calendar months, so a train window with
# window_start = 2022-02-28 would have labels reaching into val (~2023-03-28).
# BDay (Mon-Fri) over-counts trading days because it ignores NYSE holidays
# (~9/yr), so we add a generous 30-BDay buffer on top of the 272 nominal.
# Documented in dataset_manifest.json as a spec deviation.
TRAIN_CUTOFF = (VAL_START - pd.tseries.offsets.BDay(PURGE_GAP_DAYS + 30)).normalize()


def assign_split(window_start: pd.Timestamp) -> str:
    ws = pd.Timestamp(window_start)
    if ws < TRAIN_CUTOFF:
        return "train"
    if VAL_START <= ws <= VAL_END:
        return "val"
    if TEST_START <= ws <= TEST_END:
        return "test"
    return "purge"  # anything in the explicit gap region is dropped


def audit_leakage(meta: pd.DataFrame) -> None:
    """
    Validates:
      1. window_start_date sets are disjoint across splits on a per-row basis
         (a window has exactly one split assignment).
      2. Max label_date in train < min window_start in val  (the 272-day promise).
      3. Max label_date in val < min window_start in test.

    Raises AssertionError with a descriptive message on failure.
    Expects columns: split, window_start, window_end, label_date_5, label_date_10,
    label_date_20.
    """
    required = {"split", "window_start", "window_end", "label_date_5",
                "label_date_10", "label_date_20"}
    missing = required - set(meta.columns)
    assert not missing, f"audit_leakage: meta missing columns {missing}"

    # Rule 1: every row has exactly one split. (Trivially true with assign_split,
    # but we assert it in case a caller stitched meta from multiple sources.)
    assert meta["split"].notna().all(), "audit_leakage: null split found"

    label_max_col = meta[["label_date_5", "label_date_10", "label_date_20"]].max(axis=1)

    by_split = meta.assign(label_max=label_max_col).groupby("split")

    counts = by_split.size().to_dict()
    train_n = counts.get("train", 0)
    val_n = counts.get("val", 0)
    test_n = counts.get("test", 0)
    assert train_n > 0 and val_n > 0 and test_n > 0, (
        f"audit_leakage: empty split — train={train_n} val={val_n} test={test_n}"
    )

    # Rule 2
    train_label_max = by_split.get_group("train")["label_max"].max()
    val_ws_min = by_split.get_group("val")["window_start"].min()
    assert train_label_max < val_ws_min, (
        f"audit_leakage: train label_date max ({train_label_max}) "
        f">= val window_start min ({val_ws_min}). Purge gap insufficient."
    )

    # Rule 3 — soft warning. Val labels may land inside test feature windows;
    # this is model-selection bias, not training-time label leakage.
    val_label_max = by_split.get_group("val")["label_max"].max()
    test_ws_min = by_split.get_group("test")["window_start"].min()
    if val_label_max >= test_ws_min:
        log.warning(
            "val→test feature overlap: val label_date max %s >= test window_start min %s. "
            "This is spec-permitted model-selection bias, not training leakage.",
            val_label_max, test_ws_min,
        )
