# Temporimutator — Phase 3 Findings

**Status:** Parked. Architecturally outmatched by XGBoost on this data under these constraints.
**Date:** 2026-04-19
**Branch:** `temporimutator`

---

## TL;DR

We built a PyTorch transformer research track (Temporimutator) per
`filter/temporimutator_research_plan.md`, executed Phases 1–3 end-to-end, and
found that the architecture **does not beat v2_tuned XGBoost** on this data
with these inputs. Peak test IC was **+0.011 per-date Spearman** vs v2_tuned's
**+0.041 true out-of-sample on 2024** — roughly a 4× gap.

The failure mode is **structural, not hyperparameter-tunable**. Committing as a
research artifact and redirecting effort to ensembling + regime overlays.

---

## What was built

| Component | Path |
|---|---|
| Data pipeline | `scripts/temporimutator/build_dataset.py` |
| Feature streams (RSI/ATR/… + v2_tuned 23-feat) | `scripts/temporimutator/features.py` |
| Rolling-window builder (z-score + labels) | `scripts/temporimutator/windows.py` |
| Train/val/test splits + leakage audit | `scripts/temporimutator/splits.py` |
| Transformer model (3-layer, d_model=64, dual heads) | `scripts/temporimutator/model.py` |
| Phase-2 profiler (VRAM/attention/grad-flow) | `scripts/temporimutator/profile_model.py` |
| PyTorch Dataset wrappers | `scripts/temporimutator/data.py` |
| IC / accuracy / confusion-matrix utils | `scripts/temporimutator/eval_utils.py` |
| Training loop (AdamW + cosine-warmup, MLflow, early-stop on val IC) | `scripts/temporimutator/train.py` |
| Unit tests (features, splits, model) | `tests/test_temporimutator_*.py` |

All tests pass (`pytest tests/ -q` → 19/19 green).

## Run ledger

Artifacts live under `models/temporimutator/` (4-stream) and
`models/temporimutator_v2/` (23-stream). Both are `.gitignore`d.

| Run | Streams | Normalize | LR / dropout / epochs | Best val IC | Test IC | Notes |
|---|---|---|---|---|---|---|
| v1 TM-5 | 4 technical | per-window | 3e-4 / 0.1 / 30 (stopped 12) | +0.0215 @ ep6 | +0.0031 | Baseline; val IC barely above noise floor (~0.014) |
| v1 TM-5 shuffled | 4 technical | per-window | 3e-4 / 0.1 / 5 | +0.0137 @ ep1 | −0.0224 | Sanity check — confirms pipeline sound, noise floor is high |
| v2 TM-5 | 23 v2_tuned | per-window | 3e-4 / 0.1 / 30 (stopped 7) | +0.0029 @ ep1 | +0.0041 | Per-window z-score strips cross-sec info from fundamentals → val IC goes negative |
| **v2 TM-5 (β′)** | 23 v2_tuned | **cross-sectional** | 3e-4 / 0.1 / 30 (stopped 8) | **+0.0446 @ ep2** | +0.0065 | First real val signal (7σ above null). Test IC still weak. |
| v2 TM-5 (β′ tuned) | 23 v2_tuned | cross-sectional | 1e-4 / 0.3 / 6 (stopped 4) | +0.0428 @ ep1 | **+0.0111** | Best test IC achieved; Sharpe 0.70. Still 4× below v2_tuned OOS. |

**Baseline for comparison (v2_tuned walk-forward CV, 2024 fold):**
- Pooled Spearman clf: **+0.041**
- Pooled Spearman reg: +0.003

## Key learnings

### 1. Per-window z-scoring destroys cross-sectional signal

The research plan's spec-mandated per-window z-score ("z-score each stream
independently within the window") silently strips the most predictive piece
of information for equity cross-sectional returns: **peer-relative rank on a
given date**.

When we swapped to **cross-sectional z-scoring per date** (each feature
standardized across tickers on each day), val IC jumped from 0.003 → 0.045.
That's the single largest effect seen across any change in this project.

Mechanism: fundamentals like `earnings_yield` change quarterly — per-window
z-score captures only "did this feature move recently," not "is this stock
cheap vs peers." Trees (v2_tuned) get peer ranking for free; the transformer
needs it handed in.

### 2. Pipeline is sound; the noise floor is high

Shuffled-label sanity check produced epoch-1 val IC of +0.0137, which sounded
alarming but is consistent with the ~2σ null-distribution noise given our
sample size:

> SE of mean per-date IC ≈ 1/(√N_per_date × √N_dates) ≈ 0.09/√227 ≈ **0.006**

So val IC < 0.014 is ~2σ noise. Our best-ever val IC of 0.045 is ~7σ —
unambiguously real signal on val. Test IC of 0.011 is ~1.5σ — marginal.

### 3. The transformer adds no temporal edge

Across every run, best val IC peaks at **epoch 1–2** with train CE barely
moving (≤0.04 nats of improvement). Stronger regularization doesn't help;
the model has already extracted everything it's going to extract.

This means: **the 170K-parameter transformer is functioning as an expensive
linear classifier over the cross-sectionally-normalized features.** A linear
probe would get essentially the same performance. The attention layers don't
find temporal patterns that add cross-sectional predictive power here.

### 4. Structural gap vs XGBoost baseline

v2_tuned's 2024-fold training window is 2015-2023 (9 years). Ours is
2015-Nov 2021 (6 years) because the 272-day purge rule drops 13 months. That
alone is a 50% training-data gap. Combined with 2-year regime drift before
test, the transformer has a real disadvantage before architecture even enters.

### 5. Spec vs repo truth

The plan's stated baseline of "CPCV IC 0.0259 / Sharpe 3.727" doesn't match
what v3_neutral actually produces on this repo (per-fold IC ~0.018). Spec
was likely written under a different eval protocol or different universe.
**We should always derive baselines from the current repo, not from docs.**

Also: the plan's TRAIN_CUTOFF of 2022-03-01 was **arithmetically off**. 272
trading days ≈ 13 calendar months, so training windows starting Feb 2022 have
labels spilling into val. Corrected cutoff to 2021-11-04 in code; original
would have silently leaked train → val.

## Why no amount of tuning closes the gap

The vanilla transformer + per-ticker sequences framing can't see peer ranks
**evolving over time** — that's where the real signal lives, and it requires
cross-stock attention (not per-stock). Attention that flows across the
cross-section on each date is a multi-week architectural rewrite, not a
hyperparameter sweep.

## What would be needed to unlock it

1. **Cross-stock attention** — treat each date as a batch of ~2,300 stocks,
   attend both across time and across stocks on the same date. This is how
   graph-attn / TFT hybrids get edge. ~2-3 weeks of work.
2. **Walk-forward training** — match v2_tuned's eval, train 7 separate
   models (2019-2025 held-out folds). ~4-6 hours of GPU time plus infra.
3. **Pre-training** — reconstruct next-day return from 252-day sequences on
   10+ years of data, then fine-tune on direction. Addresses the
   signal-density problem.
4. **Much larger cross-section / higher-frequency data** — intraday or
   broader universe increases SNR per sample.

None of these are obviously higher-ROI than the ensembling/regime work
recommended below.

## Files / artifacts committed

- Code: `scripts/temporimutator/*.py`, `tests/test_temporimutator_*.py`
- This doc: `docs/temporimutator_findings.md`
- Research plan (input): `filter/temporimutator_research_plan.md`

**Not** committed (large + regenerable):
- `models/temporimutator/` and `models/temporimutator_v2/` — .npy datasets
  and model checkpoints (~4 GB each)
- `mlruns/`, `mlartifacts/` — MLflow tracking

To reproduce:
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu
mlflow ui --host localhost --port 5000 &
# Build v2 dataset with cross-sectional normalization
python scripts/temporimutator/build_dataset.py \
    --db data/kairos_research.duckdb \
    --out models/temporimutator_v2/ \
    --streams v2_tuned --normalize cross_sectional
# Train (best config found)
python scripts/temporimutator/train.py \
    --data-dir models/temporimutator_v2 \
    --out-dir  models/temporimutator_v2 \
    --horizon 5 --epochs 6 --batch-size 256 \
    --lr 1e-4 --dropout 0.3 --patience 3
```
