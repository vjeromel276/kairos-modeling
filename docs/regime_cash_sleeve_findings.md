# Regime-Aware Cash Sleeve — Findings (Negative Result)

**Status:** Does not improve risk-adjusted performance. Not deployed.
**Date:** 2026-04-19
**Branch:** `regime-aware-cash-sleeve`

---

## TL;DR

Implemented an opt-in `--regime-sleeve` flag on `generate_rebalance.py` that
scales equity exposure by the current `regime_history_academic` regime label.
Backtested four exposure presets (light → default) against a fully-invested
baseline on 2015-2025 weekly rebalances of the top 75 picks from
`alpha_ml_v2_tuned_clf`.

**The trade-off is nearly 1:1** — every 1pp of max drawdown saved costs ~1pp
of CAGR. Sharpe improves by at most **+0.10**, Calmar by at most **+0.10**.
Noise-level improvement on risk-adjusted metrics.

Root cause: v2_tuned_clf is a **cross-sectional** alpha signal. It works in
all regimes — long picks beat short picks in bull and bear markets. Scaling
down exposure in bearish regimes cuts alpha harvesting exactly as much as it
cuts market risk. The two cancel.

Regime-aware cash sleeves are the right lever for **market-timing** strategies,
not cross-sectional alpha. Confirmed on this dataset.

---

## What was built

| File | Purpose |
|---|---|
| `scripts/production/generate_rebalance.py` (edited) | `REGIME_EXPOSURE` map + `apply_regime_sleeve()` + `--regime-sleeve` CLI flag. Scales `portfolio_value` by regime-based exposure before trade generation. `portfolio_summary.json` now includes `regime_sleeve` block with exposure / cash split. **Opt-in only — default behavior is unchanged.** |
| `scripts/production/backtest_regime_sleeve.py` (new) | Historical simulation. For each Friday, takes top-N by alpha, equal-weights within equity sleeve, applies regime multiplier, computes 5d forward return. Supports 4 presets + custom. Writes CSV + JSON summary. |

---

## Run ledger (2015-2025, 575 weekly rebalances, 4.78M panel rows)

Universe: all tickers with `adv_20 ≥ $2M` and both alpha + 5d forward return
available. Alpha: `alpha_ml_v2_tuned_clf` (current production signal).

### Regime distribution at rebalance dates

| Regime | n | % | default exposure |
|---|---|---|---|
| high_vol_bear | 97 | 16.9% | 0.40 |
| normal_vol_neutral | 97 | 16.9% | 0.90 |
| low_vol_neutral | 88 | 15.3% | 1.00 |
| low_vol_bull | 74 | 12.9% | 1.00 |
| normal_vol_bear | 64 | 11.1% | 0.65 |
| high_vol_bull | 54 |  9.4% | 0.90 |
| normal_vol_bull | 54 |  9.4% | 1.00 |
| low_vol_bear | 29 |  5.0% | 0.75 |
| high_vol_neutral | 18 |  3.1% | 0.70 |

Average exposure under default preset: **0.812**.

### Full-period comparison

| Preset | Avg exp | CAGR | Max DD | Sharpe | Calmar | ΔCAGR | ΔMaxDD | ΔSharpe | ΔCalmar |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1.00 | +49.6% | −38.3% | +1.70 | +1.30 | — | — | — | — |
| light | 0.94 | +43.2% | −32.4% | +1.77 | +1.33 | −6.4pp | +5.9pp | +0.07 | +0.04 |
| moderate | 0.89 | +38.6% | −28.3% | +1.80 | +1.36 | −11.0pp | +10.0pp | +0.10 | +0.07 |
| bear_only | 0.85 | +35.2% | −26.4% | +1.78 | +1.33 | −14.4pp | +11.9pp | +0.08 | +0.04 |
| default | 0.81 | +31.7% | −22.7% | +1.80 | +1.40 | −17.8pp | +15.6pp | +0.10 | +0.10 |

**Trade-off:** drawdown protection is roughly proportional to CAGR lost. The
four presets span the efficient frontier without finding a free lunch.

### Per-year (default preset)

Year | baseline CAGR | baseline maxDD | sleeve CAGR | sleeve maxDD
---|---|---|---|---
2015 | +6.2% | −9.4% | +2.2% | −10.1%
2016 | +54.7% | −3.2% | +37.6% | −3.0%
2017 | +36.0% | −1.9% | +34.3% | −1.9%
2018 | +17.1% | −15.2% | +13.9% | −11.5%
2019 | +66.2% | −4.4% | +50.2% | −3.1%
**2020** | **+149.6%** | **−38.3%** | **+63.6%** | **−22.7%**
2021 | +73.3% | −7.1% | +57.8% | −6.4%
**2022** | **+6.4%** | **−16.8%** | **−2.9%** | **−13.5%**
2023 | +52.0% | −8.9% | +31.1% | −7.2%
2024 | +33.3% | −7.3% | +24.8% | −6.8%
**2025** | **+104.9%** | **−14.7%** | **+55.8%** | **−7.5%**

2020 is the clearest case: the baseline made 149% CAGR by catching COVID
recovery; the sleeve clipped it to 64% because regime labels were mostly
`high_vol_bear`. Drawdown dropped from 38% → 23% — good — but we gave up
86pp of return to save 16pp of DD. **In 2022, the sleeve turned a positive
year into a loss** (baseline +6.4% → sleeve −2.9%), a much worse outcome
than the baseline had.

---

## Root-cause analysis

The assumption behind a regime-aware cash sleeve is that the strategy's
alpha is *market-correlated* — that reducing exposure in bearish regimes
saves you from losses. For a long-only cross-sectional alpha strategy,
that assumption is wrong.

Evidence in the data:

1. **2020 baseline made +149% CAGR in a year classified mostly
   `high_vol_bear` at rebalance dates.** If alpha were correlated with
   market direction, this couldn't happen.
2. **Per-year CAGR is positive in every year for baseline.** The strategy
   doesn't have bear-market losses to protect from — it has
   bear-market *smaller gains*. Cash sleeve converts those gains to cash
   return (0%) rather than letting the alpha work.
3. **Sharpe delta maxes at +0.10 across all presets.** Risk-adjusted
   performance is essentially unchanged; we're paying for drawdown
   reduction with CAGR 1-for-1.

This is a different failure mode than the ensemble stacker — there we
learned that stacking *correlated* signals doesn't help. Here we learn
that **market-timing overlays don't help a strategy whose alpha is
uncorrelated with the market.**

---

## What this rules out

Regime-aware cash sleeves over v2_tuned_clf. No exposure preset provides
a meaningful risk-adjusted win.

The code is kept (opt-in flag, clean CLI) in case the alpha signal ever
changes to be market-correlated.

## What to try next

Two paths forward rank above this one for cross-sectional alpha:

1. **CPPI drawdown-aware allocation** — the `idol-objective` branch
   (commit `4f95e82`) already implemented this. Reported Calmar +1.76
   vs baseline +1.56 — a bigger risk-adjusted win than the best regime
   preset here (+1.40). CPPI sizes based on realized drawdown, which is
   the right framing: cut risk when actually losing money, not when a
   macro regime classifier predicts we might.
2. **Beta/factor hedging** — if the portfolio's net beta is high,
   shorting SPY in proportion eliminates market risk without touching
   alpha. A proper beta calc on the existing top-75 picks would tell us
   whether this is worth doing. Cheap to prototype.
3. **Tail hedges** — OTM SPY puts during high_vol regimes only.
   Explicit tail protection without cash drag.

Recommendation: merge CPPI from `idol-objective`, or rebuild it here and
combine with beta hedging.

---

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Sweep all presets on 2015-2025 v2_tuned_clf
python scripts/production/backtest_regime_sleeve.py \
    --db data/kairos.duckdb \
    --start 2015-01-01 --end 2025-12-31 \
    --alpha alpha_ml_v2_tuned_clf \
    --top-n 75 --preset all

# Live use (opt-in)
python scripts/production/generate_rebalance.py \
    --db data/kairos.duckdb --date 2026-04-17 \
    --from-alpaca --regime-sleeve \
    --prior-holdings outputs/rebalance/2026-04-10/picks.csv
```

Summary JSON at `scripts/ml/outputs/backtest_regime_sleeve_summary.json`.
