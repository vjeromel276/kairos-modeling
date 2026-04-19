"""
Feature computation for Temporimutator input streams.

All functions are pure: pandas Series in, pandas Series out. No I/O.
Formulas follow temporimutator_research_plan.md (lines 138-162).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi_wilder(close: pd.Series, n: int = 14) -> pd.Series:
    """RSI with Wilder smoothing (alpha = 1/n, adjust=False)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def volume_ratio(volume: pd.Series, n: int = 20) -> pd.Series:
    """volume / rolling_n_avg(volume)."""
    return volume / volume.rolling(n).mean()


def trend_extension(close: pd.Series, n: int = 50) -> pd.Series:
    """close / EMA(n) - 1. Uses pandas ewm(span=n, adjust=False) to match spec."""
    ema = close.ewm(span=n, adjust=False).mean()
    return close / ema - 1


def atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR(n) with Wilder smoothing applied to true range."""
    prev_close = close.shift(1)
    hl = high - low
    hpc = (high - prev_close).abs()
    lpc = (low - prev_close).abs()
    tr = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return atr


def atr_ratio(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """ATR(n) / close — dimensionless volatility feature stream."""
    return atr_wilder(high, low, close, n) / close


def vol_scalar(close: pd.Series, window: int = 252) -> pd.Series:
    """
    Std of daily log returns over the trailing `window` days.

    Returns a Series aligned with `close`; value at index i is computed over
    close[i-window+1 .. i]. Appended outside the transformer as an absolute-
    volatility signal for the force head.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window).std()


def compute_all_streams(df_ticker: pd.DataFrame) -> pd.DataFrame:
    """
    Given a per-ticker dataframe sorted by date with columns
    [high, low, closeadj, volume], append the 4 feature streams and the
    vol_scalar column. Returns a new dataframe (does not mutate input).
    """
    out = df_ticker.copy()
    out["rsi"] = rsi_wilder(out["closeadj"])
    out["vol_ratio"] = volume_ratio(out["volume"])
    out["trend_ext"] = trend_extension(out["closeadj"])
    out["atr_ratio"] = atr_ratio(out["high"], out["low"], out["closeadj"])
    out["vol_scalar"] = vol_scalar(out["closeadj"])
    return out


STREAM_COLS = ("rsi", "vol_ratio", "trend_ext", "atr_ratio")
