#!/usr/bin/env python3
"""
Temporimutator Phase 2 profiling script.

Runs:
  - forward-pass shape sanity check on random data
  - VRAM profile at batch sizes 128, 256, 512 (training step = fwd + bwd + opt)
  - parameter count
  - gradient-flow audit (every param gets a non-zero gradient)
  - attention capture for 3 sample stocks drawn from val_sequences.npy
  - writes model_config.json
  - logs params / metrics / artifacts to MLflow experiment temporimutator_architecture

Usage:
  python scripts/temporimutator/profile_model.py \\
      --data-dir models/temporimutator/ \\
      --out-dir  models/temporimutator/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_THIS = Path(__file__).resolve()
_PROJECT_ROOT = _THIS.parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.temporimutator.model import TMConfig, TemporimutatorModel  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("temporimutator.profile")


def forward_shape_check(device: torch.device) -> dict:
    torch.manual_seed(0)
    m = TemporimutatorModel().to(device).eval()
    B = 8
    seqs = torch.randn(B, 252, 4, device=device)
    vs = torch.rand(B, 1, device=device) * 0.05
    with torch.no_grad():
        dir_logits, force = m(seqs, vs)
    assert dir_logits.shape == (B, 3), dir_logits.shape
    assert force.shape == (B,), force.shape
    probs = F.softmax(dir_logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, device=device), atol=1e-5)
    return {"direction_logits_shape": list(dir_logits.shape), "force_shape": list(force.shape)}


def vram_profile(device: torch.device, batch_sizes=(128, 256, 512)) -> dict:
    if device.type != "cuda":
        log.warning("No CUDA device — skipping VRAM profile.")
        return {}
    results = {}
    for B in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            m = TemporimutatorModel().to(device).train()
            opt = torch.optim.AdamW(m.parameters(), lr=3e-4)

            seqs = torch.randn(B, 252, 4, device=device)
            vs = torch.rand(B, 1, device=device) * 0.05
            y_dir = torch.randint(0, 3, (B,), device=device)
            y_force = torch.rand(B, device=device)

            # Warm start (one step) + timed step
            for _ in range(2):
                opt.zero_grad(set_to_none=True)
                dir_logits, force = m(seqs, vs)
                loss = F.cross_entropy(dir_logits, y_dir) + F.mse_loss(force, y_force)
                loss.backward()
                opt.step()

            torch.cuda.synchronize()
            peak_mib = torch.cuda.max_memory_allocated() / (1024 ** 2)
            t0 = time.time()
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                dir_logits, force = m(seqs, vs)
                loss = F.cross_entropy(dir_logits, y_dir) + F.mse_loss(force, y_force)
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            step_ms = (time.time() - t0) / 5 * 1000
            results[B] = {"peak_mib": round(peak_mib, 1), "step_ms": round(step_ms, 2)}
            log.info("  batch=%d  peak_VRAM=%.1f MiB  step=%.2f ms", B, peak_mib, step_ms)
            del m, opt, seqs, vs, y_dir, y_force, dir_logits, force, loss
        except torch.cuda.OutOfMemoryError as e:
            results[B] = {"error": "OOM", "msg": str(e)[:200]}
            log.warning("  batch=%d  OOM", B)
        torch.cuda.empty_cache()
    return results


def gradient_flow_audit(device: torch.device) -> dict:
    torch.manual_seed(0)
    m = TemporimutatorModel().to(device).train()
    B = 8
    seqs = torch.randn(B, 252, 4, device=device)
    vs = torch.rand(B, 1, device=device) * 0.05
    y_dir = torch.randint(0, 3, (B,), device=device)
    y_force = torch.rand(B, device=device)
    dir_logits, force = m(seqs, vs)
    loss = F.cross_entropy(dir_logits, y_dir) + F.mse_loss(force, y_force)
    loss.backward()

    zero_params = []
    grad_norms = {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            zero_params.append(name)
            continue
        g = p.grad.detach().abs().sum().item()
        if g == 0.0:
            zero_params.append(name)
        grad_norms[name] = g
    return {
        "n_params_with_nonzero_grad": len(grad_norms) - len(zero_params),
        "n_params_total_trainable": len(grad_norms),
        "params_with_zero_grad": zero_params[:10],
    }


def attention_capture(device: torch.device, data_dir: Path, n_stocks: int = 3) -> dict:
    val_seqs = np.load(data_dir / "val_sequences.npy", mmap_mode="r")
    val_scalars = np.load(data_dir / "val_scalars.npy", mmap_mode="r")

    rng = np.random.default_rng(7)
    idxs = rng.choice(len(val_seqs), size=n_stocks, replace=False)
    seqs = torch.from_numpy(np.asarray(val_seqs[idxs])).to(device)
    vs = torch.from_numpy(np.asarray(val_scalars[idxs])).to(device)

    torch.manual_seed(0)
    m = TemporimutatorModel().to(device).eval()
    with torch.no_grad():
        dir_logits, force, attn_list = m(seqs, vs, return_attn=True)

    # Save to npy: (layers, n_stocks, heads, seq, seq)
    stacked = torch.stack(attn_list, dim=0).cpu().numpy()  # (L, B, H, S, S)
    np.save(data_dir / "phase2_attention_sample.npy", stacked)

    # Summary statistics per layer
    per_layer = []
    for L_i, w in enumerate(attn_list):
        # w: (B, H, S, S); entropy over last dim
        probs = w.clamp_min(1e-12)
        ent = -(probs * probs.log()).sum(dim=-1)  # (B, H, S)
        per_layer.append({
            "layer": L_i,
            "mean_attn_entropy": float(ent.mean().item()),
            "max_attn_weight": float(w.max().item()),
            "min_attn_weight": float(w.min().item()),
        })
    return {
        "sampled_val_idxs": idxs.tolist(),
        "attention_stats_per_layer": per_layer,
        "attention_saved": str(data_dir / "phase2_attention_sample.npy"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Temporimutator Phase 2 profiler")
    p.add_argument("--data-dir", type=Path, default=Path("models/temporimutator"),
                   help="Directory containing val_sequences.npy / val_scalars.npy")
    p.add_argument("--out-dir", type=Path, default=Path("models/temporimutator"),
                   help="Where model_config.json is written")
    p.add_argument("--mlflow-uri", default="http://localhost:5000",
                   help="MLflow tracking URI (empty to disable)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)
    if device.type == "cuda":
        log.info("  CUDA: %s  |  %s",
                 torch.version.cuda, torch.cuda.get_device_name(0))

    cfg = TMConfig()
    report: dict = {"config": cfg.to_dict()}

    log.info("Forward-pass shape check...")
    report["forward_shape"] = forward_shape_check(device)
    log.info("  ok")

    log.info("Parameter count...")
    m = TemporimutatorModel(cfg).to(device)
    report["param_count"] = m.count_parameters()
    log.info("  total=%d  trainable=%d", report["param_count"]["total"],
             report["param_count"]["trainable"])
    del m

    log.info("VRAM profile...")
    report["vram"] = vram_profile(device)

    log.info("Gradient flow audit...")
    report["gradient_flow"] = gradient_flow_audit(device)
    log.info("  %d / %d trainable params have non-zero gradient",
             report["gradient_flow"]["n_params_with_nonzero_grad"],
             report["gradient_flow"]["n_params_total_trainable"])

    log.info("Attention capture...")
    if (args.data_dir / "val_sequences.npy").exists():
        report["attention"] = attention_capture(device, args.data_dir)
        log.info("  saved to %s", report["attention"]["attention_saved"])
    else:
        log.warning("  val_sequences.npy not found in %s — skipping", args.data_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = args.out_dir / "model_config.json"
    with open(cfg_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Wrote %s", cfg_path)

    if args.mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment("temporimutator_architecture")
            with mlflow.start_run(run_name="phase2_profile"):
                mlflow.log_params(cfg.to_dict())
                mlflow.log_metric("param_count_total", report["param_count"]["total"])
                mlflow.log_metric("param_count_trainable", report["param_count"]["trainable"])
                for B, info in report.get("vram", {}).items():
                    if "peak_mib" in info:
                        mlflow.log_metric(f"vram_peak_mib_b{B}", info["peak_mib"])
                        mlflow.log_metric(f"step_ms_b{B}", info["step_ms"])
                if "attention" in report:
                    for rec in report["attention"]["attention_stats_per_layer"]:
                        mlflow.log_metric(
                            f"attn_entropy_layer{rec['layer']}",
                            rec["mean_attn_entropy"],
                        )
                mlflow.log_artifact(str(cfg_path))
                if "attention" in report:
                    mlflow.log_artifact(report["attention"]["attention_saved"])
            log.info("MLflow run logged")
        except Exception as e:
            log.warning("MLflow logging failed: %s", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
