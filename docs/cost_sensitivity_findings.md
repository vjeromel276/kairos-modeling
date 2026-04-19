# Cost Sensitivity — Full Verification

**Status:** CPPI-production underperforms baseline at realistic costs.
**Winner under costs:** `dd_linear_20_30` across all book sizes < $10M.
**Date:** 2026-04-19
**Branch:** `idol-objective`

---

## TL;DR

Three verifications under honest cost accounting:

1. **Spread sensitivity** (0→30 bps): CPPI-prod break-even vs baseline is
   **~12 bps**. Below 12 bps spread CPPI wins; above 12 bps baseline wins.
   Realistic universe-weighted cost on a $2M+ ADV mid-cap portfolio is
   **likely 12-20 bps**, putting CPPI-prod on or past the break-even line.

2. **Impact model** (scales with book size): Impact is negligible at
   $100k-$1M books. Bites at $10M+. At $100M, impact dominates and
   **low-exposure CPPI (`cppi_85_3`, avg 43% alloc) becomes optimal**
   because smaller trade sizes → less impact amplification.

3. **CPPI (floor, multiplier) grid under 15 bps:** the gross-optimal
   `85/6.67` drops to middle of the pack. **Net-optimal CPPI is
   `78/4.55_pk100`** (looser floor, smaller multiplier, same 100%
   peak exposure) — but even this ties dd_linear at Calmar 1.21,
   doesn't beat it.

**Net conclusion:** at current book size ($100k) and realistic costs,
**`dd_linear_20_30` strictly dominates the current production CPPI
config** on net Calmar (+0.09), net Sharpe (+0.09), and net CAGR (+2.2pp).

---

## 1. Spread sensitivity — CPPI-prod vs baseline

**Net Calmar** as round-trip spread rises:

| spread | full_exposure | cppi_85_667_PROD | Δ (CPPI − base) | winner |
|---|---|---|---|---|
| 0 bps | 1.515 | 1.726 | **+0.21** | CPPI |
| 5 bps | 1.389 | 1.497 | +0.11 | CPPI |
| 10 bps | 1.267 | 1.308 | +0.04 | CPPI |
| **12 bps (est)** | **~1.22** | **~1.22** | **~0.00** | **tie** |
| 15 bps | 1.149 | 1.129 | −0.02 | baseline |
| 20 bps | 1.035 | 0.964 | −0.07 | baseline |
| 30 bps | 0.818 | 0.700 | −0.12 | baseline |

Break-even ≈ 12 bps. For a universe filtered at `adv_20 ≥ $2M` the
mid-cap average round-trip cost is typically 15-25 bps, which puts
CPPI-prod in the losing zone under honest costs.

## 2. Strategy rankings at 15 bps (no impact)

**Net Calmar:**

| Rank | Strategy | Net Calmar | Net CAGR | Net Sharpe | Annual Turnover | Annual Cost |
|---|---|---|---|---|---|---|
| 1 | dd_linear_15_50 | 1.22 | 33.6% | 1.345 | 28.6× | 7.9% |
| 2 | **dd_linear_20_30** | **1.22** | **33.6%** | **1.344** | 28.5× | 7.9% |
| 3 | dd_linear_25_20 | 1.21 | 33.6% | 1.345 | 28.7× | 7.9% |
| 4 | cppi_90_3 | 1.21 | 33.6% | 1.343 | 8.8× | 7.9% |
| 5 | cppi_85_3 | 1.20 | 33.6% | 1.344 | 13.2× | 7.8% |
| 6 | cppi_80_3 | 1.20 | 33.6% | 1.344 | 17.6× | 7.8% |
| 7 | dd_convex_20_30 | 1.15 | 33.3% | 1.333 | 30.1× | 7.8% |
| 8 | **full_exposure** (baseline) | **1.15** | **33.0%** | **1.322** | 31.0× | 7.1% |
| 9 | **cppi_85_667_PROD** | **1.13** | **31.4%** | **1.256** | 25.0× | 8.2% |
| 10 | dd_regime_hybrid | 1.11 | 32.9% | 1.317 | 28.5× | 7.5% |

**CPPI-prod ranks 9th of 10** on net Calmar. Baseline (no allocation overlay)
is 8th. **`dd_linear_20_30` leads by +0.09 Calmar, +2.2pp CAGR** vs prod.

Note `cppi_90_3` appears competitive, but its avg allocation is 28% — it's
essentially holding ~70% cash and using vol-target leverage to hit 25%
target vol. That's not a real win, it's a leverage trick.

## 3. CPPI grid under 15 bps

Top 5 by net Calmar (from peak-lock grid, re-run with costs):

| Strategy | Floor | Mult | Peak | Net Calmar | Net CAGR |
|---|---|---|---|---|---|
| **cppi_78_4.55_pk100** | 0.78 | 4.55 | 1.00 | **1.21** | 33.3% |
| cppi_80_5.00_pk100 | 0.80 | 5.00 | 1.00 | 1.19 | 33.0% |
| cppi_82_5.56_pk100 | 0.82 | 5.56 | 1.00 | 1.17 | 32.6% |
| **cppi_85_6.67_pk100 (PROD)** | **0.85** | **6.67** | 1.00 | **1.13** | **31.4%** |
| cppi_88_8.33_pk100 | 0.88 | 8.33 | 1.00 | 1.06 | 29.8% |

**Under costs, the optimal CPPI floor shifts from 85% → 78%** (looser floor
= larger cushion = less frequent cutting = less trading). Production's
85/6.67 is 3rd-to-last of 5 net of costs.

Even the best CPPI (78/4.55) only ties dd_linear at 1.21 — dd_linear
gets there with a simpler mechanism.

## 4. Impact model — book-size sensitivity

Impact cost = `impact_coef × delta × (delta × pv / adv)`. At `impact_coef=0.1`:

**Net CAGR by book size (spread=15 + impact=0.1):**

| Strategy | $100k | $1M | $10M | $100M |
|---|---|---|---|---|
| **dd_linear_20_30** | **33.6%** | **33.5%** | 32.8% | 25.7% |
| cppi_85_3 | 33.6% | 33.5% | 33.2% | **30.0%** |
| full_exposure | 33.0% | 32.9% | 32.2% | 25.5% |
| cppi_85_667_PROD | 31.4% | 31.3% | 30.6% | 24.4% |

At **$100k-$1M**: dd_linear_20_30 wins.
At **$10M**: dd_linear and cppi_85_3 are neck and neck.
At **$100M+**: cppi_85_3 dominates — low avg allocation (43%) means small
trade sizes, minimal impact, best net of costs.

Current Kairos is at ~$100k → **dd_linear_20_30 is correct for today's
book. If/when scale crosses $10M, reconsider toward a cash-heavier CPPI.**

---

## What this means for production

### The honest comparison

At today's book size and 15 bps assumed cost:

| vs baseline (full_exposure) | CPPI-prod | **dd_linear_20_30** |
|---|---|---|
| Δ Calmar | **−0.02** (worse) | **+0.07** (better) |
| Δ CAGR | **−1.6pp** (worse) | **+0.6pp** (better) |
| Δ Sharpe | **−0.07** (worse) | **+0.02** (better) |
| Δ Max DD | +0.9pp (better) | +1.1pp (better) |

**The current `generate_rebalance.py` default is defensibly the wrong
choice if costs are ≥12 bps.** Baseline (no CPPI) beats it; dd_linear
beats both.

### Three possible production changes

Ranked by caution:

1. **Disable CPPI by default** (`--no-cppi` becomes default behavior).
   Keep code path available for users who expect different cost regimes
   or want CPPI anyway. Minimal risk — just reverts to baseline.

2. **Implement `dd_linear_20_30` as a new allocation mode** alongside
   CPPI, opt-in via `--allocation-mode dd_linear`. Deploy after paper
   trading validation.

3. **Ship cost-aware backtest as a first-class tool** so every future
   "X beats baseline" claim lives under honest accounting, not gross.

Before making any change, we should:
- Validate the cost model against a few weeks of live Alpaca fills
  (actual realized slippage tells us the true cost assumption)
- Paper-trade dd_linear in parallel with CPPI for 4-8 weeks
- Compare live net performance before switching defaults

**I'd recommend starting with #3** (commit this cost suite) + a **backlog
ticket** for #1 / #2 pending live validation. Flipping a default is a
production decision, not a backtest decision.

---

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Full 3-part suite (spread sweep + impact + CPPI grid under costs)
python scripts/evaluation/cost_sensitivity_suite.py \
    --db data/kairos.duckdb \
    --output-dir outputs/evaluation/cost_suite

# Focused impact scaling test
# (reuse backtest function directly at multiple portfolio-values)
```

Results saved to `outputs/evaluation/cost_suite/cost_sensitivity_all_results.csv`
(131 rows tagged by regime, strategy, spread, impact).
