# Transaction Cost Model — Initial Findings

**Status:** CPPI production config is **not net-optimal** under realistic costs.
**Date:** 2026-04-19
**Branch:** `idol-objective`

---

## TL;DR

Added per-trade transaction costs to `backtest_capital_allocation.py` (spread
model + optional impact). Re-ran the full strategy sweep at 15 bps round-trip.

**Under honest cost accounting the ranking changes materially:**

| Strategy | Gross Calmar | **Net Calmar** | Gross CAGR | **Net CAGR** |
|---|---|---|---|---|
| **dd_linear_20_30** | 1.65 | **1.22** | 44.5% | **33.6%** |
| dd_linear_15_50 | 1.65 | 1.22 | 44.5% | 33.6% |
| cppi_90_3 | 1.62 | 1.20 | 44.4% | 33.6% |
| full_exposure (baseline) | 1.52 | 1.15 | 42.8% | 33.0% |
| **cppi_85_667_PROD** | **1.63** | **1.13** | 43.6% | **31.4%** |

- **CPPI production underperforms baseline** on both net Calmar and net CAGR.
- `dd_linear_20_30` is the new net-optimal: +0.07 Calmar and +0.6pp CAGR vs baseline.
- Gross winners (CPPI) become net losers because vol-targeting amplifies costs.

---

## What was built

Modified `scripts/evaluation/backtest_capital_allocation.py`:

1. **Position-space turnover tracking.** Per rebalance: compute
   `position = weight × allocation`, measure `|position_new − position_prev|.sum()`.
   Captures both shape changes and CPPI allocation changes. First period
   includes the initial portfolio build-up. Cash transitions include full
   liquidation turnover.

2. **Cost deduction.** `cost = turnover × spread_bps / 10_000`, subtracted
   from the raw period return BEFORE vol-target scaling. When vol-target
   applies a leverage factor, cost scales with it too (realistic — leverage
   amplifies dollar trading volumes).

3. **Optional linear impact model.** `impact_bps_per_trade =
   impact_coef × trade_value / adv_20`. Requires a portfolio-value
   assumption. Not used in the headline run — conservative baseline is
   spread-only. Enabled via `--impact-coef`.

4. **CLI flags:** `--spread-bps`, `--impact-coef`, `--portfolio-value`.

5. **Extended output columns:** `annual_turnover`, `annual_cost`,
   `sharpe_gross`, `ann_return_gross`, `calmar_gross` — net and gross
   side-by-side for every strategy.

---

## Why CPPI flipped from winner to loser

Two mechanics combine to amplify CPPI's costs:

1. **Allocation changes trade.** Every time CPPI scales exposure (a 5%
   drawdown move might cut exposure from 100% → 67%), the portfolio must
   sell 33% of its positions. CPPI's protection buys you lower drawdowns
   but at the price of constant reallocation trades.

2. **Vol-target leverages.** CPPI's dampened returns have lower realized
   vol. To hit the 25% target vol, the backtest multiplies by a larger
   scale factor — ~1.5× more leverage than baseline. That leverage
   amplifies BOTH returns and costs, but costs scale linearly with
   leverage while returns don't always (leveraged trades also lose more
   on the losers).

Evidence:

| Strategy | Annual Turnover | Annual Cost | Cost per Unit Turnover |
|---|---|---|---|
| full_exposure | 31.0× | 7.1% | ≈ 23 bps |
| cppi_85_667_PROD | 25.0× | 8.2% | ≈ 33 bps |
| dd_linear_20_30 | 28.5× | 7.9% | ≈ 28 bps |

CPPI has ~20% lower turnover but ~44% higher cost-per-unit-turnover, net:
higher total cost despite less trading.

---

## Why dd_linear wins

- Avg allocation 92% (vs CPPI 80%) → less vol-target leverage needed
- Linear scaling only cuts during actual drawdowns (like CPPI) but **less
  aggressively** — fewer allocation-change trades
- Min allocation 46% (vs CPPI prod min 15%) — never forces a deep
  liquidation

Net effect: comparable drawdown protection with far less cost amplification.

---

## Three things to verify before changing production

1. **Spread sensitivity.** Run the sweep at 5/10/15/20/30 bps to find
   the break-even cost level where CPPI ties baseline. Gives us a
   defensible "assumes at least X bps" footnote.
2. **Add impact.** Linear impact at `coef=0.1` on a $100k book, scaling
   at larger sizes. Likely widens the CPPI loss further.
3. **Re-run (floor, multiplier) CPPI grid under costs.** The prior grid
   was gross-only. A lower-turnover CPPI config might retain enough
   drawdown benefit to win on net.

Also worth reconsidering: the vol-target interaction. The ex-post
vol-scaling in the backtest isn't fully realistic — in deployment you'd
size positions based on an ex-ante vol estimate, not retroactively
leverage. That changes the cost calculus. Worth a separate experiment.

---

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Gross (what we had before)
python scripts/evaluation/backtest_capital_allocation.py \
    --db data/kairos.duckdb --spread-bps 0 \
    --output-dir outputs/evaluation/capital_allocation_gross

# Net, 15 bps round-trip (headline run)
python scripts/evaluation/backtest_capital_allocation.py \
    --db data/kairos.duckdb --spread-bps 15 \
    --output-dir outputs/evaluation/capital_allocation_net15

# Net + impact (adds ~1-2% more drag for large portfolios)
python scripts/evaluation/backtest_capital_allocation.py \
    --db data/kairos.duckdb --spread-bps 15 \
    --impact-coef 0.1 --portfolio-value 100000 \
    --output-dir outputs/evaluation/capital_allocation_net15_impact
```
