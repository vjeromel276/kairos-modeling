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

from scripts.temporimutator.features import STREAM_COLS  # noqa: E402
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
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    universe: list[str] | None = None
    if universe_csv is not None:
        universe = pd.read_csv(universe_csv)["ticker"].astype(str).str.upper().tolist()
        log.info("Universe filter loaded: %d tickers", len(universe))

    log.info("Opening %s", db_path)
    conn = duckdb.connect(str(db_path), read_only=True)
    df = load_sep(conn, universe)
    conn.close()
    log.info(
        "Loaded sep_base_academic: rows=%d tickers=%d dates %s → %s",
        len(df),
        df["ticker"].nunique(),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    # -------- Pass 1: build raw windows per ticker --------
    t0 = time.time()
    raw_windows: list = []
    tickers_seen = 0
    tickers_skipped_short = 0
    tickers_with_windows = 0
    for ticker, g in df.groupby("ticker", sort=False):
        tickers_seen += 1
        if len(g) < min_ticker_rows:
            tickers_skipped_short += 1
            continue
        g = g.assign(ticker=ticker)
        rws = build_raw_windows(g)
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

        seqs = np.empty((n, WINDOW, len(STREAM_COLS)), dtype=np.float32)
        scalars = np.empty((n, 1), dtype=np.float32)
        for i, j in enumerate(idxs):
            rw = raw_windows[int(j)]
            seqs[i] = zscore_sequence(rw.sequence)
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
    args = p.parse_args()

    mlflow_uri = args.mlflow_uri if args.mlflow_uri else None
    build(args.db, args.out, args.universe, args.min_ticker_rows, mlflow_uri)
    return 0


if __name__ == "__main__":
    sys.exit(main())
