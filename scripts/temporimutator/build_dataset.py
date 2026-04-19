#!/usr/bin/env python3
"""
Temporimutator — Phase 1 dataset builder.

Reads sep_base_academic from a DuckDB research snapshot, computes the 4 input
streams plus vol_scalar per ticker, constructs 252-day rolling windows with
21-day step, assigns cross-sectional direction thresholds per (label_date,
horizon), winsorizes force labels on the train split, applies a 272-day
purge chronological split, and writes numpy + parquet artifacts to disk.

Usage:
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
  python scripts/temporimutator/build_dataset.py \\
      --db data/kairos_research.duckdb \\
      --out models/temporimutator/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# Package-relative imports — works when invoked as `python -m scripts.temporimutator.build_dataset`
# or when this file's parent dirs are on sys.path. We add project root to sys.path to cover the
# common `python scripts/temporimutator/build_dataset.py` invocation.
_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.temporimutator.features import (  # noqa: E402
    STREAM_COLS,
    STREAM_PRESETS,
    V2_TUNED_SOURCES,
)
from scripts.temporimutator.splits import (  # noqa: E402
    TEST_END,
    TEST_START,
    TRAIN_CUTOFF,
    VAL_END,
    VAL_START,
    assign_split,
    audit_leakage,
)
from scripts.temporimutator.windows import (  # noqa: E402
    HORIZONS,
    STEP,
    WINDOW,
    build_raw_windows,
    direction_from_threshold,
    zscore_sequence,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("temporimutator.build_dataset")


def load_sep(conn: duckdb.DuckDBPyConnection, universe: list[str] | None) -> pd.DataFrame:
    q = """
        SELECT ticker, date, high, low, closeadj, volume
        FROM sep_base_academic
        WHERE closeadj IS NOT NULL
          AND high IS NOT NULL
          AND low IS NOT NULL
          AND volume IS NOT NULL
        ORDER BY ticker, date
    """
    df = conn.execute(q).fetchdf()
    if universe is not None:
        df = df[df["ticker"].isin(universe)].reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    return df


def join_v2_tuned_features(
    conn: duckdb.DuckDBPyConnection, df: pd.DataFrame
) -> pd.DataFrame:
    """Left-join the 5 v2_tuned feat_* tables onto (ticker, date)."""
    out = df
    for table, cols in V2_TUNED_SOURCES.items():
        col_list = ", ".join(cols)
        log.info("  joining %s (%d cols)", table, len(cols))
        feat = conn.execute(
            f"SELECT ticker, date, {col_list} FROM {table}"
        ).fetchdf()
        feat["date"] = pd.to_datetime(feat["date"])
        out = out.merge(feat, on=["ticker", "date"], how="left")
    # Forward-fill per ticker so a sparsely-updating fundamental doesn't create
    # 90%+ NaN gaps inside a window (fundamentals update quarterly).
    log.info("  forward-filling fundamental columns per ticker")
    cols_to_ffill = [c for cols in V2_TUNED_SOURCES.values() for c in cols]
    out[cols_to_ffill] = out.groupby("ticker")[cols_to_ffill].ffill()
    return out


def cross_sectional_zscore(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    Per-date z-score across tickers. Replaces each feature value with its
    cross-sectional standardized score on that date, preserving peer-rank
    information that per-window temporal z-scoring would destroy.

    Robust to NaN (filled with 0 after division) and to dates with near-zero
    cross-sectional std (std floored at 1e-8).
    """
    log.info("  cross-sectional z-scoring %d columns per date", len(cols))
    grouped = df.groupby("date", sort=False)
    for col in cols:
        mean = grouped[col].transform("mean")
        std = grouped[col].transform("std").replace(0, np.nan)
        z = (df[col] - mean) / std
        df[col] = z.fillna(0.0).astype(np.float32)
    return df


def write_parquet_via_duckdb(df: pd.DataFrame, out_path: Path) -> None:
    """
    Writes a DataFrame to Parquet via DuckDB to avoid the pyarrow/duckdb
    "file-scheme already registered" collision that breaks `pd.to_parquet`
    when duckdb has already been imported in the process.
    """
    tmp = duckdb.connect(":memory:")
    try:
        tmp.register("_df", df)
        tmp.execute(f"COPY (SELECT * FROM _df) TO '{out_path}' (FORMAT PARQUET)")
    finally:
        tmp.close()


def file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT, stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def build(
    db_path: Path,
    out_dir: Path,
    universe_csv: Path | None,
    min_ticker_rows: int,
    mlflow_uri: str | None,
    streams: str = "technical",
    normalize: str = "per_window",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    if streams not in STREAM_PRESETS:
        raise ValueError(f"Unknown --streams preset '{streams}'. Options: {list(STREAM_PRESETS)}")
    stream_cols = STREAM_PRESETS[streams]
    log.info("Stream preset: %s (%d streams)", streams, len(stream_cols))

    universe: list[str] | None = None
    if universe_csv is not None:
        universe = pd.read_csv(universe_csv)["ticker"].astype(str).str.upper().tolist()
        log.info("Universe filter loaded: %d tickers", len(universe))

    log.info("Opening %s", db_path)
    conn = duckdb.connect(str(db_path), read_only=True)
    df = load_sep(conn, universe)
    if streams == "v2_tuned":
        df = join_v2_tuned_features(conn, df)
    conn.close()

    if normalize == "cross_sectional":
        # Pre-normalize feature streams per date across tickers. Skips the
        # per-window z-score step in the materialization loop below.
        df = cross_sectional_zscore(df, list(stream_cols))
    elif normalize != "per_window":
        raise ValueError(f"Unknown --normalize '{normalize}'")
    log.info(
        "Loaded sep_base_academic: rows=%d tickers=%d dates %s → %s  cols=%d",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
        df.shape[1],
    )

    # -------- Pass 1: build raw windows per ticker --------
    t0 = time.time()
    raw_windows: list = []
    tickers_seen = 0
    tickers_skipped_short = 0
    tickers_with_windows = 0
    # For v2_tuned we still need vol_scalar (for force) so we keep
    # compute_technical=True (it also computes RSI etc but they're not in
    # stream_cols so they're just carried and ignored).
    for ticker, g in df.groupby("ticker", sort=False):
        tickers_seen += 1
        if len(g) < min_ticker_rows:
            tickers_skipped_short += 1
            continue
        g = g.assign(ticker=ticker)
        rws = build_raw_windows(g, stream_cols=stream_cols)
        if rws:
            tickers_with_windows += 1
            raw_windows.extend(rws)
    log.info(
        "Pass 1 done in %.1fs — tickers seen=%d, skipped(short)=%d, with_windows=%d, raw_windows=%d",
        time.time() - t0,
        tickers_seen,
        tickers_skipped_short,
        tickers_with_windows,
        len(raw_windows),
    )
    if not raw_windows:
        raise RuntimeError("No windows produced. Check universe and source table.")

    # -------- Build meta dataframe --------
    meta_rows = []
    for i, rw in enumerate(raw_windows):
        row = {
            "idx": i,
            "ticker": rw.ticker,
            "window_start": rw.window_start,
            "window_end": rw.window_end,
            "vol_scalar": rw.vol_scalar,
        }
        for h in HORIZONS:
            row[f"ret_{h}d"] = rw.forward_returns[h]
            row[f"label_date_{h}"] = rw.label_dates[h]
        meta_rows.append(row)
    meta = pd.DataFrame(meta_rows)
    meta["split"] = meta["window_start"].apply(assign_split)

    n_purge = int((meta["split"] == "purge").sum())
    meta = meta[meta["split"] != "purge"].reset_index(drop=True)
    log.info(
        "Split assignment: train=%d val=%d test=%d (dropped purge=%d)",
        (meta["split"] == "train").sum(),
        (meta["split"] == "val").sum(),
        (meta["split"] == "test").sum(),
        n_purge,
    )

    # -------- Cross-sectional direction thresholds per (label_date, horizon) --------
    # spec: threshold = median_abs_return / 2 across the cross-section on that label date
    thresholds: dict[int, pd.Series] = {}
    for h in HORIZONS:
        s = meta[[f"label_date_{h}", f"ret_{h}d"]].copy()
        s["abs_ret"] = s[f"ret_{h}d"].abs()
        thresholds[h] = s.groupby(f"label_date_{h}")["abs_ret"].median() / 2.0

    for h in HORIZONS:
        thr = meta[f"label_date_{h}"].map(thresholds[h])
        dirs = [
            direction_from_threshold(ret, t if not pd.isna(t) else 0.0)
            for ret, t in zip(meta[f"ret_{h}d"].to_numpy(), thr.to_numpy())
        ]
        meta[f"direction_{h}"] = dirs
        meta[f"force_{h}"] = (meta[f"ret_{h}d"].abs() / meta["vol_scalar"]).astype(np.float32)

    # -------- Force winsorization fit on train, applied to all --------
    force_caps: dict[int, float] = {}
    for h in HORIZONS:
        train_force = meta.loc[meta["split"] == "train", f"force_{h}"].to_numpy()
        cap = float(np.nanquantile(train_force, 0.99))
        force_caps[h] = cap
        meta[f"force_{h}"] = meta[f"force_{h}"].clip(upper=cap)

    # -------- Leakage audit --------
    audit_leakage(meta)
    log.info("Leakage audit passed.")

    # -------- Materialize arrays per split --------
    manifest: dict = {
        "stream_preset": streams,
        "stream_cols": list(stream_cols),
        "n_streams": len(stream_cols),
        "normalize": normalize,
        "window": WINDOW,
        "step": STEP,
        "horizons": list(HORIZONS),
        "purge_gap_days": 272,
        "train_cutoff": str(TRAIN_CUTOFF.date()),
        "val_range": [str(VAL_START.date()), str(VAL_END.date())],
        "test_range": [str(TEST_START.date()), str(TEST_END.date())],
        "force_winsor_caps_p99_train": force_caps,
        "universe_csv": str(universe_csv) if universe_csv else None,
        "db_path": str(db_path),
        "db_sha1": file_sha1(db_path) if db_path.stat().st_size < 5_000_000_000 else "skipped_large",
        "git_sha": git_sha(),
        "splits": {},
    }

    for split in ("train", "val", "test"):
        sub = meta[meta["split"] == split].reset_index(drop=True)
        idxs = sub["idx"].to_numpy()
        n = len(idxs)
        log.info("Writing split=%s  n=%d", split, n)
        if n == 0:
            continue

        seqs = np.empty((n, WINDOW, len(stream_cols)), dtype=np.float32)
        scalars = np.empty((n, 1), dtype=np.float32)
        do_per_window_zscore = normalize == "per_window"
        for i, j in enumerate(idxs):
            rw = raw_windows[int(j)]
            if do_per_window_zscore:
                seqs[i] = zscore_sequence(rw.sequence)
            else:
                # Already cross-sectionally normalized; just clean NaN/inf.
                s = rw.sequence.astype(np.float32, copy=True)
                s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
                seqs[i] = s
            scalars[i, 0] = rw.vol_scalar

        np.save(out_dir / f"{split}_sequences.npy", seqs)
        np.save(out_dir / f"{split}_scalars.npy", scalars)
        for h in HORIZONS:
            np.save(
                out_dir / f"{split}_labels_dir_{h}d.npy",
                sub[f"direction_{h}"].to_numpy(dtype=np.int8),
            )
            np.save(
                out_dir / f"{split}_labels_force_{h}d.npy",
                sub[f"force_{h}"].to_numpy(dtype=np.float32),
            )

        meta_cols = (
            ["ticker", "window_start", "window_end", "vol_scalar"]
            + [f"label_date_{h}" for h in HORIZONS]
            + [f"ret_{h}d" for h in HORIZONS]
            + [f"direction_{h}" for h in HORIZONS]
            + [f"force_{h}" for h in HORIZONS]
        )
        write_parquet_via_duckdb(sub[meta_cols], out_dir / f"{split}_meta.parquet")

        manifest["splits"][split] = {
            "n": int(n),
            "class_balance_5": {
                str(k): int(v)
                for k, v in sub["direction_5"].value_counts().sort_index().items()
            },
            "class_balance_10": {
                str(k): int(v)
                for k, v in sub["direction_10"].value_counts().sort_index().items()
            },
            "class_balance_20": {
                str(k): int(v)
                for k, v in sub["direction_20"].value_counts().sort_index().items()
            },
            "force_5_percentiles": {
                p: float(np.quantile(sub["force_5"], p / 100))
                for p in (50, 90, 99)
            },
            "date_range": [str(sub["window_start"].min().date()), str(sub["window_start"].max().date())],
        }

    with open(out_dir / "dataset_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log.info("Wrote %s/dataset_manifest.json", out_dir)

    # -------- MLflow logging --------
    if mlflow_uri:
        try:
            import mlflow

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment("temporimutator_data_pipeline")
            with mlflow.start_run(run_name="phase1_build"):
                mlflow.log_params({
                    "window": WINDOW,
                    "step": STEP,
                    "horizons": ",".join(str(h) for h in HORIZONS),
                    "purge_gap_days": 272,
                    "train_cutoff": manifest["train_cutoff"],
                    "val_start": manifest["val_range"][0],
                    "test_start": manifest["test_range"][0],
                    "universe_csv": manifest["universe_csv"] or "all",
                    "git_sha": manifest["git_sha"],
                })
                for split, info in manifest["splits"].items():
                    mlflow.log_metric(f"n_{split}", info["n"])
                    for h in HORIZONS:
                        cb = info[f"class_balance_{h}"]
                        for cls, cnt in cb.items():
                            mlflow.log_metric(f"n_{split}_dir{h}_cls{cls}", cnt)
                for h, cap in force_caps.items():
                    mlflow.log_metric(f"force_cap_p99_train_{h}d", cap)
                mlflow.log_artifact(str(out_dir / "dataset_manifest.json"))
            log.info("MLflow run logged to %s", mlflow_uri)
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description="Build Temporimutator Phase 1 dataset")
    p.add_argument("--db", required=True, type=Path, help="Path to DuckDB research snapshot")
    p.add_argument("--out", required=True, type=Path, help="Output directory")
    p.add_argument("--universe", type=Path, default=None,
                   help="Optional CSV of tickers to restrict universe")
    p.add_argument("--min-ticker-rows", type=int, default=WINDOW + max(HORIZONS) + 50,
                   help="Skip tickers with fewer rows than this")
    p.add_argument("--mlflow-uri", default="http://localhost:5000",
                   help="MLflow tracking URI (or empty string to disable)")
    p.add_argument("--streams", choices=list(STREAM_PRESETS), default="technical",
                   help="Which feature-stream preset to build (technical|v2_tuned)")
    p.add_argument("--normalize", choices=("per_window", "cross_sectional"),
                   default="per_window",
                   help="Z-score within each 252-day window, or cross-sectionally per date")
    args = p.parse_args()

    mlflow_uri = args.mlflow_uri if args.mlflow_uri else None
    build(args.db, args.out, args.universe, args.min_ticker_rows, mlflow_uri,
          streams=args.streams, normalize=args.normalize)
    return 0


if __name__ == "__main__":
    sys.exit(main())
