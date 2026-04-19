"""
Rolling-window construction for Temporimutator.

Each window is a 252-trading-day slice of the 4 feature streams, z-scored
in-window, with a separate vol_scalar and forward-return labels for multiple
horizons. Windows are emitted in two passes:

  Pass 1: collect raw forward returns per (ticker, window_end, horizon).
  Pass 2: the caller computes cross-sectional direction thresholds globally
          (per label_date × horizon) and re-calls `label_windows()`.

This two-pass shape is enforced because direction = sign relative to the
cross-sectional median absolute return on the label date (spec line 184-186).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from typing import Sequence

from .features import STREAM_COLS, compute_all_streams

WINDOW = 252
STEP = 21
HORIZONS = (5, 10, 20)
MAX_NAN_FRAC = 0.10

# A 252-trading-day window should span ~365 calendar days. Cap at 400 to tolerate
# holidays and minor gaps but reject delisted/reassigned tickers whose bar history
# jumps years — otherwise the split-by-window_start heuristic breaks.
MAX_WINDOW_SPAN_DAYS = 400
# Likewise bound the forward-label span. 20 trading days is ~28 calendar days;
# reject if > 60 to catch tickers with late-window trading halts or delistings.
MAX_LABEL_SPAN_DAYS = 60


@dataclass
class RawWindow:
    ticker: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    sequence: np.ndarray   # (252, 4) raw streams (not yet z-scored)
    vol_scalar: float      # scalar
    # forward close and ret at each horizon; label_dates indexed by horizon
    label_dates: dict      # {h: Timestamp}
    forward_returns: dict  # {h: float}


def build_raw_windows(
    df_ticker: pd.DataFrame,
    window: int = WINDOW,
    step: int = STEP,
    horizons: Iterable[int] = HORIZONS,
    max_nan_frac: float = MAX_NAN_FRAC,
    stream_cols: Sequence[str] = STREAM_COLS,
    compute_technical: bool = True,
) -> list[RawWindow]:
    """
    Given a per-ticker dataframe with columns [date, high, low, closeadj, volume]
    (plus any extra feature columns already merged in), return a list of
    RawWindow records. Drops windows with >max_nan_frac NaN in any stream.
    Sequences are raw (not yet z-scored).

    `stream_cols` selects which columns of the resulting dataframe become the
    sequence streams. `compute_technical=True` appends the 4 technical streams
    (rsi/vol_ratio/trend_ext/atr_ratio) and vol_scalar; disable when you don't
    need them and they're not in `stream_cols`.
    """
    df = df_ticker.sort_values("date").reset_index(drop=True)
    if len(df) < window + max(horizons):
        return []

    feat = compute_all_streams(df) if compute_technical else df.copy()
    # vol_scalar is required regardless of stream choice (force label needs it)
    if "vol_scalar" not in feat.columns:
        from .features import vol_scalar as _vs
        feat["vol_scalar"] = _vs(feat["closeadj"])
    ticker = feat["ticker"].iloc[0] if "ticker" in feat.columns else df_ticker.iloc[0]["ticker"]
    max_h = max(horizons)

    out: list[RawWindow] = []
    last_end_idx = len(feat) - max_h - 1  # need max_h forward bars for labels
    # window indices: [start, start+window)
    for start in range(0, last_end_idx - window + 1 + 1, step):
        end = start + window  # exclusive; last index inside window is end-1
        if end - 1 > last_end_idx:
            break

        sub = feat.iloc[start:end]
        seq = sub[list(stream_cols)].to_numpy(dtype=np.float32)
        # NaN fraction per stream
        nan_frac = np.isnan(seq).mean(axis=0)
        if (nan_frac > max_nan_frac).any():
            continue

        # vol_scalar aligned to the window's last day
        vs = feat["vol_scalar"].iloc[end - 1]
        if pd.isna(vs) or vs <= 0:
            continue

        window_start = pd.Timestamp(feat["date"].iloc[start])
        window_end = pd.Timestamp(feat["date"].iloc[end - 1])

        # Reject windows whose calendar span is wildly off the expected 252 trading days.
        # Catches delisted + reassigned symbols with multi-year gaps in their history.
        if (window_end - window_start).days > MAX_WINDOW_SPAN_DAYS:
            continue

        close_end = feat["closeadj"].iloc[end - 1]
        label_dates: dict = {}
        forward_returns: dict = {}
        skip = False
        for h in horizons:
            forward_idx = end - 1 + h
            if forward_idx >= len(feat):
                skip = True
                break
            close_fwd = feat["closeadj"].iloc[forward_idx]
            if pd.isna(close_fwd) or close_fwd <= 0 or close_end <= 0:
                skip = True
                break
            label_date = pd.Timestamp(feat["date"].iloc[forward_idx])
            if (label_date - window_end).days > MAX_LABEL_SPAN_DAYS:
                skip = True
                break
            ret = float(close_fwd / close_end - 1.0)
            label_dates[h] = label_date
            forward_returns[h] = ret
        if skip:
            continue

        out.append(
            RawWindow(
                ticker=str(ticker),
                window_start=window_start,
                window_end=window_end,
                sequence=seq,
                vol_scalar=float(vs),
                label_dates=label_dates,
                forward_returns=forward_returns,
            )
        )

    return out


def zscore_sequence(seq: np.ndarray) -> np.ndarray:
    """Per-stream z-score of a (T, F) array. NaNs are filled with 0 after z-scoring."""
    mean = np.nanmean(seq, axis=0, keepdims=True)
    std = np.nanstd(seq, axis=0, keepdims=True)
    z = (seq - mean) / (std + 1e-8)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return z.astype(np.float32, copy=False)


def direction_from_threshold(ret: float, threshold: float) -> int:
    """0=down, 1=flat, 2=up. Threshold is absolute return (flat half-width)."""
    if ret > threshold:
        return 2
    if ret < -threshold:
        return 0
    return 1


def compute_cross_sectional_thresholds(
    meta: pd.DataFrame, horizons: Iterable[int]
) -> dict[int, pd.Series]:
    """
    Per spec line 184-186: threshold = (median abs return on that label_date) / 2.

    Returns dict[horizon] → Series indexed by label_date with the threshold.
    Callers map each window's label_date to its threshold for that horizon.
    """
    out: dict[int, pd.Series] = {}
    for h in horizons:
        col_ret = f"ret_{h}d"
        col_date = f"label_date_{h}"
        sub = meta[[col_date, col_ret]].dropna()
        abs_ret = sub[col_ret].abs()
        median_by_date = abs_ret.groupby(sub[col_date]).median()
        out[h] = median_by_date / 2.0
    return out
