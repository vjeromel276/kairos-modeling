"""
Evaluation utilities for Temporimutator.

Primary metric is the Information Coefficient (IC): cross-sectional Spearman
correlation between the signal and the raw forward return, computed per
label-date, then averaged. This matches the evaluation methodology of
v3_neutral (spec lines 270-273).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def direction_signal(probs: np.ndarray) -> np.ndarray:
    """P(up) - P(down) given softmax probs (N, 3) with class order [down, flat, up]."""
    return probs[:, 2] - probs[:, 0]


def ic_per_date(signal: np.ndarray, ret: np.ndarray, dates: np.ndarray) -> pd.Series:
    """Spearman IC of `signal` vs `ret` per unique date in `dates`. Returns Series."""
    df = pd.DataFrame({"signal": signal, "ret": ret, "date": dates})
    out = {}
    for dt, g in df.groupby("date"):
        if len(g) < 5:
            continue
        if g["signal"].std() == 0 or g["ret"].std() == 0:
            continue
        s = g["signal"].rank(method="average")
        r = g["ret"].rank(method="average")
        if s.std() == 0 or r.std() == 0:
            continue
        out[dt] = float(np.corrcoef(s, r)[0, 1])
    return pd.Series(out).sort_index()


def ic_summary(signal: np.ndarray, ret: np.ndarray, dates: np.ndarray) -> dict:
    ser = ic_per_date(signal, ret, dates)
    if len(ser) == 0:
        return {"ic_mean": float("nan"), "ic_std": float("nan"),
                "ic_sharpe": float("nan"), "n_dates": 0}
    mean = float(ser.mean())
    std = float(ser.std(ddof=1)) if len(ser) > 1 else float("nan")
    sharpe = mean / std * np.sqrt(252) if std and not np.isnan(std) and std > 0 else float("nan")
    return {
        "ic_mean": mean,
        "ic_std": std,
        "ic_sharpe": sharpe,
        "n_dates": int(len(ser)),
        "ic_pos_frac": float((ser > 0).mean()),
    }


def direction_accuracy(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Argmax accuracy given probs (N, 3) and integer labels (N,)."""
    preds = probs.argmax(axis=1)
    return float((preds == y_true).mean())


def confusion_matrix(probs: np.ndarray, y_true: np.ndarray, n_classes: int = 3) -> np.ndarray:
    preds = probs.argmax(axis=1)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, preds):
        cm[int(t), int(p)] += 1
    return cm


def force_mae(force_pred: np.ndarray, force_true: np.ndarray) -> float:
    return float(np.abs(force_pred - force_true).mean())
