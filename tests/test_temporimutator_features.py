"""Unit tests for scripts/temporimutator/features.py."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.temporimutator.features import (
    atr_wilder,
    rsi_wilder,
    trend_extension,
    vol_scalar,
    volume_ratio,
)


def test_rsi_constant_series_is_nan_or_neutral():
    # Constant price → zero gain and loss → RSI undefined (NaN) under Wilder.
    close = pd.Series([100.0] * 50)
    r = rsi_wilder(close, n=14)
    # After warmup period, values should be NaN (0/0) or missing.
    tail = r.iloc[14:]
    assert tail.isna().all(), "RSI on constant series should be NaN (0/0)"


def test_rsi_monotonic_up_is_100():
    # Strictly increasing → no losses → RSI = 100.
    close = pd.Series(np.arange(1, 61, dtype=float))
    r = rsi_wilder(close, n=14).dropna()
    assert (r.round(6) == 100.0).all(), "Monotonic up RSI must equal 100"


def test_rsi_monotonic_down_is_zero():
    close = pd.Series(np.arange(60, 0, -1, dtype=float))
    r = rsi_wilder(close, n=14).dropna()
    assert (r.round(6) == 0.0).all(), "Monotonic down RSI must equal 0"


def test_volume_ratio_constant_equals_one():
    vol = pd.Series([1000.0] * 30)
    vr = volume_ratio(vol, n=20).dropna()
    assert np.allclose(vr.to_numpy(), 1.0)


def test_trend_extension_constant_equals_zero():
    close = pd.Series([50.0] * 100)
    te = trend_extension(close, n=50)
    # With adjust=False and constant input, EMA equals the constant immediately.
    assert np.allclose(te.to_numpy(), 0.0, atol=1e-12)


def test_atr_basic_shape_and_nonneg():
    rng = np.random.default_rng(42)
    n = 100
    close = pd.Series(100 + rng.standard_normal(n).cumsum())
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    atr = atr_wilder(high, low, close, n=14).dropna()
    assert len(atr) > 0
    assert (atr >= 0).all()


def test_atr_flat_market_is_zero_after_warmup():
    # high=low=close → true range 0 → ATR 0 post-warmup.
    close = pd.Series([100.0] * 30)
    high = close.copy()
    low = close.copy()
    atr = atr_wilder(high, low, close, n=14)
    assert (atr.iloc[14:].round(10) == 0.0).all()


def test_vol_scalar_window_shape_and_nonneg():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(rng.standard_normal(300).cumsum() * 0.01))
    vs = vol_scalar(close, window=252).dropna()
    assert len(vs) == 300 - 252  # 48 tail values
    assert (vs >= 0).all()


def test_vol_scalar_constant_is_zero():
    close = pd.Series([100.0] * 300)
    vs = vol_scalar(close, window=252).dropna()
    # log returns are all 0 → std = 0
    assert np.allclose(vs.to_numpy(), 0.0)
