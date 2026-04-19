#!/usr/bin/env python3
"""
Temporimutator Phase 3 — training loop.

Trains TemporimutatorModel for a chosen label horizon (5/10/20 days).
AdamW + cosine LR with warmup, inverse-frequency weighted cross-entropy
on direction + MSE on force, early stopping on val IC (not loss), best
checkpoint saved by IC. Evaluates on test set at end. Logs everything
to MLflow experiment temporimutator_training.

Usage (TM-5):
  python scripts/temporimutator/train.py \\
      --data-dir models/temporimutator \\
      --out-dir  models/temporimutator \\
      --horizon 5 \\
      --epochs 30 \\
      --batch-size 256
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.temporimutator.data import TMDataset, inverse_freq_weights  # noqa: E402
from scripts.temporimutator.eval_utils import (  # noqa: E402
    confusion_matrix,
    direction_accuracy,
    direction_signal,
    force_mae,
    ic_summary,
)
from scripts.temporimutator.model import TMConfig, TemporimutatorModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("temporimutator.train")

# Baselines for TM-5 comparison (spec says plan-stated vs actual repo-measured).
BASELINES = {
    5: {"plan_stated_ic": 0.0259, "plan_stated_ic_sharpe": 3.727,
        "actual_v3_neutral_ic": 0.0178, "actual_v3_neutral_ic_sharpe": 0.625},
}


def cosine_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_eval(model, loader, meta_df, device, horizon):
    model.eval()
    all_probs = []
    all_force = []
    all_ydir = []
    all_yforce = []
    all_idx = []
    with torch.no_grad():
        for seq, vs, y_dir, y_force, idx in loader:
            seq = seq.to(device, non_blocking=True)
            vs = vs.to(device, non_blocking=True)
            dir_logits, force = model(seq, vs)
            probs = F.softmax(dir_logits, dim=-1)
            all_probs.append(probs.cpu().numpy())
            all_force.append(force.cpu().numpy())
            all_ydir.append(y_dir.numpy())
            all_yforce.append(y_force.numpy())
            all_idx.append(idx.numpy())
    probs = np.concatenate(all_probs)
    force_pred = np.concatenate(all_force)
    y_dir = np.concatenate(all_ydir)
    y_force = np.concatenate(all_yforce)
    idx = np.concatenate(all_idx)

    # Lookup raw returns + label_date for ICs
    meta_ret = meta_df[f"ret_{horizon}d"].to_numpy()
    meta_date = meta_df[f"label_date_{horizon}"].to_numpy()
    ret = meta_ret[idx]
    dates = meta_date[idx]

    signal = direction_signal(probs)
    ic = ic_summary(signal, ret, dates)
    acc = direction_accuracy(probs, y_dir)
    cm = confusion_matrix(probs, y_dir)
    mae = force_mae(force_pred, y_force)
    return {
        "ic": ic,
        "direction_accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "force_mae": mae,
        "n_samples": int(len(y_dir)),
    }, probs, force_pred


def read_meta(data_dir: Path, split: str) -> pd.DataFrame:
    # Avoid pyarrow — use duckdb to read parquet (same workaround as build_dataset)
    import duckdb
    con = duckdb.connect(":memory:")
    try:
        path = data_dir / f"{split}_meta.parquet"
        df = con.execute(f"SELECT * FROM '{path}'").fetchdf()
    finally:
        con.close()
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="Train a Temporimutator horizon")
    p.add_argument("--data-dir", type=Path, default=Path("models/temporimutator"))
    p.add_argument("--out-dir", type=Path, default=Path("models/temporimutator"))
    p.add_argument("--horizon", type=int, choices=(5, 10, 20), default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-pct", type=float, default=0.05)
    p.add_argument("--force-weight", type=float, default=0.1,
                   help="Loss weight on force MSE relative to weighted CE")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.1,
                   help="Transformer dropout rate (also applies to head sublayers)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--mlflow-uri", default="http://localhost:5000")
    p.add_argument("--run-name", default=None)
    p.add_argument("--shuffle-train-labels", action="store_true",
                   help="Sanity check: permute train labels. Val IC should hover near 0.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    H = args.horizon

    train_ds = TMDataset(args.data_dir, "train", H)
    val_ds = TMDataset(args.data_dir, "val", H)
    test_ds = TMDataset(args.data_dir, "test", H)
    n_streams = int(train_ds._sequences.shape[-1])
    log.info("Sizes — train=%d  val=%d  test=%d  n_streams=%d",
             len(train_ds), len(val_ds), len(test_ds), n_streams)

    if args.shuffle_train_labels:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(train_ds))
        # mmap arrays are read-only; materialize writable copies before permuting
        train_ds._dir = np.array(train_ds._dir)[perm]
        train_ds._force = np.array(train_ds._force)[perm]
        log.warning("SHUFFLE-TRAIN-LABELS enabled — val IC should be ~0 in a working pipeline.")

    train_meta = read_meta(args.data_dir, "train")
    val_meta = read_meta(args.data_dir, "val")
    test_meta = read_meta(args.data_dir, "test")

    # Class weights from train directions
    cw = inverse_freq_weights(np.asarray(train_ds._dir))
    log.info("Class weights (down/flat/up): %s", cw.tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = TemporimutatorModel(TMConfig(n_streams=n_streams, dropout=args.dropout)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cw_t = torch.from_numpy(cw).to(device)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_pct))
    log.info("Training steps: %d  warmup: %d  force_weight=%.2f",
             total_steps, warmup_steps, args.force_weight)

    # MLflow
    mlflow_run = None
    if args.mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("temporimutator_training")
            mlflow_run = mlflow.start_run(
                run_name=args.run_name or f"tm{H}_e{args.epochs}_b{args.batch_size}_lr{args.lr}"
            )
            mlflow.log_params({
                "horizon": H, "epochs": args.epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": args.weight_decay,
                "warmup_pct": args.warmup_pct, "force_weight": args.force_weight,
                "patience": args.patience, "seed": args.seed,
                "n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
                "class_weights": str(cw.tolist()),
            })
            if H in BASELINES:
                mlflow.log_params({f"baseline_{k}": v for k, v in BASELINES[H].items()})
        except Exception as e:
            log.warning("MLflow init failed: %s", e)
            mlflow_run = None

    best_ic = -float("inf")
    best_epoch = -1
    patience_left = args.patience
    history = []
    step = 0
    t_start = time.time()
    best_path = args.out_dir / f"tm{H}_best.pt"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        ep_loss_ce = 0.0
        ep_loss_mse = 0.0
        ep_n = 0
        t_ep = time.time()
        for seq, vs, y_dir, y_force, _ in train_loader:
            lr_now = cosine_warmup(step, total_steps, warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now

            seq = seq.to(device, non_blocking=True)
            vs = vs.to(device, non_blocking=True)
            y_dir = y_dir.to(device, non_blocking=True)
            y_force = y_force.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            dir_logits, force = model(seq, vs)
            loss_ce = F.cross_entropy(dir_logits, y_dir, weight=cw_t)
            loss_mse = F.mse_loss(force, y_force)
            loss = loss_ce + args.force_weight * loss_mse
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = seq.size(0)
            ep_loss_ce += loss_ce.item() * bsz
            ep_loss_mse += loss_mse.item() * bsz
            ep_n += bsz
            step += 1

        ep_loss_ce /= max(1, ep_n)
        ep_loss_mse /= max(1, ep_n)
        t_train = time.time() - t_ep

        t_val0 = time.time()
        val_report, _, _ = run_eval(model, val_loader, val_meta, device, H)
        t_val = time.time() - t_val0

        ic_mean = val_report["ic"]["ic_mean"]
        log.info(
            "epoch %2d/%d  train_ce=%.4f  train_mse=%.4f  val_ic=%.4f  val_acc=%.3f  "
            "val_mae=%.3f  t_train=%.1fs  t_val=%.1fs",
            epoch + 1, args.epochs, ep_loss_ce, ep_loss_mse,
            ic_mean, val_report["direction_accuracy"], val_report["force_mae"],
            t_train, t_val,
        )

        history.append({
            "epoch": epoch + 1,
            "lr": lr_now,
            "train_ce": ep_loss_ce,
            "train_mse": ep_loss_mse,
            "val_ic": ic_mean,
            "val_ic_sharpe": val_report["ic"]["ic_sharpe"],
            "val_acc": val_report["direction_accuracy"],
            "val_force_mae": val_report["force_mae"],
            "t_train_s": t_train,
            "t_val_s": t_val,
        })

        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics({
                "train_ce": ep_loss_ce,
                "train_mse": ep_loss_mse,
                "val_ic": ic_mean,
                "val_ic_sharpe": val_report["ic"]["ic_sharpe"],
                "val_ic_pos_frac": val_report["ic"]["ic_pos_frac"],
                "val_acc": val_report["direction_accuracy"],
                "val_force_mae": val_report["force_mae"],
                "lr": lr_now,
            }, step=epoch + 1)

        if ic_mean > best_ic:
            best_ic = ic_mean
            best_epoch = epoch + 1
            patience_left = args.patience
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "config": model.cfg.to_dict(),
                "val_ic": best_ic,
                "horizon": H,
            }, best_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stop at epoch %d (no val IC improvement for %d epochs).",
                         epoch + 1, args.patience)
                break

    log.info("Training complete in %.1fs. Best val_ic=%.4f @ epoch %d",
             time.time() - t_start, best_ic, best_epoch)

    # Reload best and evaluate on test
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_report, test_probs, test_force = run_eval(model, test_loader, test_meta, device, H)

    log.info("TEST  ic=%.4f  ic_sharpe=%.3f  acc=%.3f  mae=%.3f  n_dates=%d",
             test_report["ic"]["ic_mean"],
             test_report["ic"]["ic_sharpe"],
             test_report["direction_accuracy"],
             test_report["force_mae"],
             test_report["ic"]["n_dates"])
    if H in BASELINES:
        b = BASELINES[H]
        dta = test_report["ic"]["ic_mean"] - b["actual_v3_neutral_ic"]
        dtp = test_report["ic"]["ic_mean"] - b["plan_stated_ic"]
        log.info("Baseline deltas — vs actual_v3_neutral_ic=%+.4f  vs plan_stated_ic=%+.4f",
                 dta, dtp)

    final_report = {
        "horizon": H,
        "best_epoch": best_epoch,
        "best_val_ic": best_ic,
        "test": test_report,
        "baselines": BASELINES.get(H),
        "history": history,
        "run_time_s": time.time() - t_start,
    }
    report_path = args.out_dir / f"tm{H}_training_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2, default=str)
    log.info("Wrote %s", report_path)

    # Save test predictions for later ensemble work
    np.save(args.out_dir / f"tm{H}_test_probs.npy", test_probs.astype(np.float32))
    np.save(args.out_dir / f"tm{H}_test_force.npy", test_force.astype(np.float32))

    if mlflow_run is not None:
        import mlflow
        mlflow.log_metrics({
            "test_ic": test_report["ic"]["ic_mean"],
            "test_ic_sharpe": test_report["ic"]["ic_sharpe"],
            "test_ic_pos_frac": test_report["ic"]["ic_pos_frac"],
            "test_acc": test_report["direction_accuracy"],
            "test_force_mae": test_report["force_mae"],
            "best_epoch": best_epoch,
            "best_val_ic": best_ic,
        })
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(best_path))
        mlflow.end_run()

    return 0


if __name__ == "__main__":
    sys.exit(main())
