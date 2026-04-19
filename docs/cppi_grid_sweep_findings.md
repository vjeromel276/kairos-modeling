# CPPI 2D Grid Sweep — Findings

**Status:** Production config (85% floor, 6.67x multiplier) verified as Calmar-optimal on 2015-2025.
**Date:** 2026-04-19
**Branch:** `idol-objective`

---

## TL;DR

Ran a 2D grid sweep around the production CPPI config to map the local
optimum on the (floor_pct, multiplier) surface. **Production config
(85/6.67) is already at the Calmar peak** — 1.73 vs 1.52 baseline.
No free lunch in the grid.

The local surface has a shallow efficient frontier at `peak_exposure=1.0`
(i.e., `multiplier × (1 - floor) = 1.0`):
  - Lower floors (78-82%) give slightly higher Sharpe/CAGR, worse DD, lower Calmar
  - Higher floors (85%) give lower Sharpe but lowest DD, best Calmar
  - Tighter floors (88%+) actively destroy both
  - "Overclocked" CPPI (peak > 1.0) is universally worse

---

## What was run

21-config grid (7 floors × 3 peak-exposure targets), holding exposure-at-peak
constant so multiplier is tied to floor:

| floor | peak=1.00 mult | peak=1.25 mult | peak=1.50 mult |
|---|---|---|---|
| 0.78 | 4.55 | 5.68 | 6.82 |
| 0.80 | 5.00 | 6.25 | 7.50 |
| 0.82 | 5.56 | 6.94 | 8.33 |
| **0.85** | **6.67 (prod)** | 8.33 | 10.00 |
| 0.88 | 8.33 | 10.42 | 12.50 |
| 0.90 | 10.00 | 12.50 | 15.00 |
| 0.92 | 12.50 | 15.63 | 18.75 |

Baseline (full exposure) + production config pinned for reference.

## Top 10 results (sorted by Calmar)

| Strategy | Floor | Mult | Peak | Sharpe | CAGR | MaxDD | Calmar | AvgAlloc |
|---|---|---|---|---|---|---|---|---|
| **cppi_85_6.67 (PROD)** | 0.85 | 6.67 | 1.00 | +1.743 | +43.6% | −25.3% | **+1.73** | 86% |
| cppi_82_5.56 | 0.82 | 5.56 | 1.00 | +1.774 | +44.3% | −26.0% | +1.71 | 90% |
| cppi_80_5.00 | 0.80 | 5.00 | 1.00 | +1.780 | +44.5% | −26.3% | +1.69 | 92% |
| cppi_78_4.55 | 0.78 | 4.55 | 1.00 | +1.782 | +44.5% | −26.6% | +1.68 | 93% |
| cppi_88_8.33 | 0.88 | 8.33 | 1.00 | +1.644 | +41.1% | −25.5% | +1.61 | 78% |
| cppi_80_6.25 | 0.80 | 6.25 | 1.25 | +1.750 | +43.8% | −27.5% | +1.59 | 96% |
| cppi_78_5.68 | 0.78 | 5.68 | 1.25 | +1.757 | +43.9% | −27.8% | +1.58 | 97% |
| cppi_82_6.94 | 0.82 | 6.94 | 1.25 | +1.738 | +43.5% | −27.5% | +1.58 | 93% |
| cppi_80_7.50 | 0.80 | 7.50 | 1.50 | +1.749 | +43.7% | −28.3% | +1.54 | 98% |
| **baseline_full** | — | — | — | +1.711 | +42.8% | −28.2% | +1.51 | 100% |

Bottom ranks (for reference): every tight-floor config with multiplier ≥12
produced Calmar < 1.00 — worse than baseline by a lot.

## Interpretation

1. **Production config sits on the knee.** The surface is smooth around
   it; nearby points trade Sharpe ↔ DD almost 1:1 but Calmar barely moves.
2. **Sweet spot is `floor ∈ [80, 85]` with `peak_exposure = 1.0`.** Outside
   this band, either exposure gets too low at peak (tight floors) or the
   cuts come too late during drawdowns (loose floors, high peak-target).
3. **No evidence for "smarter" CPPI here.** The simple shape
   `exposure = mult × cushion`, capped at 1.0, is doing its job.

## Recommendation

**Do not change the production CPPI parameters.** The grid search validates
the hand-picked 85/6.67 as near-optimal by Calmar. Any alternative requires
accepting a tradeoff (marginally higher Sharpe for marginally higher DD, or
vice versa) that isn't clearly worth it.

Next steps don't live in this parameter space — they live in:
  - **Transaction cost accounting** (gross backtest ignores ~30% weekly turnover × 15-30 bps)
  - **Hold-period / turnover-cap sweep** (under honest costs)
  - **Orthogonal features** or **beta hedging**

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Peak-lock grid (21 configs, 7 floors × 3 peak targets)
python scripts/evaluation/cppi_grid_sweep.py \
    --db data/kairos.duckdb --mode peak_lock \
    --output-dir outputs/evaluation/cppi_grid

# Free grid (25 configs, floor × mult independently — for completeness)
python scripts/evaluation/cppi_grid_sweep.py \
    --db data/kairos.duckdb --mode free \
    --output-dir outputs/evaluation/cppi_grid
```

Results in `outputs/evaluation/cppi_grid/` (gitignored):
  - `cppi_grid_peak_lock.csv`
  - `cppi_grid_peak_lock_summary.json`
