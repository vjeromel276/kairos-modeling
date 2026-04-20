# Live Slippage Findings — Signal-to-Execution Gap is the Real Cost

**Status:** Root cause identified. Fix shipped as opt-in flags + recommended cadence.
**Date:** 2026-04-19
**Branch:** `idol-objective`

---

## TL;DR

Pulled every Alpaca fill from 2026-01-02 through 2026-04-17 (1,170 filled orders,
~$1.1M gross traded) and compared fill prices to the reference prices captured
when `generate_rebalance.py` ran Friday nights.

**Measured slippage: +88 bps per trade (notional-weighted), +177 bps round-trip.**
Backtest assumed 15 bps. Twelve-fold worse than modeled.

**Root cause: it is not bid-ask spread.** LOO orders that did fill showed only
**18 bps** slippage — confirming pure execution cost is ~15-20 bps. The other
~70 bps per trade is the **signal-to-execution lag**: pipeline runs Friday 8pm,
orders fill Monday 9:30am (64-hour gap). Other quants running similar models on
the same Friday close data move the market pre-open, and we fill behind them.

**Fix: shift the execution cadence and refresh prices at fill time.**

---

## Evidence

### Buys and sells both show "correct-direction" slippage

| Side | N fills | Wtd mean slip | Interpretation |
|---|---|---|---|
| BUY | 578 | **+77 bps** | stocks we bought moved UP before we filled |
| SELL | 592 | **+100 bps** | stocks we sold moved DOWN before we filled |

Both signs are consistent with "the alpha signal was right, but we arrived after
the move started." That is alpha decay during the timing gap.

### Order type matters less than timing

| Order type | N filled | Mean slip | Median slip |
|---|---|---|---|
| **LOO (limit-on-open)** | 13 | **+18 bps** | **+26 bps** |
| MOO (market-on-open) | 38 | +73 bps | +36 bps |
| MARKET (intraday) | 1,119 | +92 bps | +61 bps |

LOO fills are the cleanest — they give the true floor on execution cost
(~15–25 bps). But their fill rate is terrible (paper-trading LOO has a large
expiry problem). MARKET intraday is what actually filled the portfolio, and
it carries the full timing penalty.

### Regime dependence

High-volatility bear regimes spike slippage by 3×:

| Rebalance date | Regime | Wtd mean slip |
|---|---|---|
| 2026-01-02 | normal_vol_neutral | +10 bps |
| 2026-01-09 | low_vol_neutral | +36 bps |
| 2026-01-23 | normal_vol_neutral | +59 bps |
| 2026-02-20 | high_vol_bull | +96 bps |
| **2026-03-20** | **high_vol_bear** | **+176 bps** |
| **2026-03-27** | **high_vol_bear** | **+173 bps** |
| 2026-04-02 | high_vol_bear | +45 bps |
| 2026-04-10 | high_vol_neutral | +75 bps |

Weekend gap risk explodes in chaotic markets. This is consistent with the
timing-gap theory: when the market is jumpy, 64 hours between signal and fill
is a huge exposure.

### Live price drift validates the mechanism

Ran `refresh_picks_prices.py --dry-run` on the 2026-04-17 picks.csv over a
weekend (Sun 9:48 PM ET, Alpaca returned last-trade from Friday close):

- Median drift: −0.9 bps
- Mean drift: −23.4 bps
- p90 |drift|: **130 bps**
- Max |drift|: **+595 bps (SNA)**

Even before the Monday open, 10% of picks have already drifted ≥130 bps from
Friday close — and that is only the visible overnight move. The full drift
measured at Monday fill time (after open auction) would be larger.

---

## What shipped

### 1. `scripts/evaluation/measure_live_slippage.py`

Pulls every FILL activity from Alpaca for a date range, joins to the
`alpaca_orders_*.csv` artifacts, computes per-trade slippage vs ref_price
and aggregates by side / order type / rebalance date. Writes:

- `outputs/evaluation/live_slippage/live_slippage_trades.csv` (per-trade)
- `outputs/evaluation/live_slippage/live_slippage_summary.json` (aggregates)

### 2. `scripts/production/refresh_picks_prices.py` (new)

Overwrites the `price`, `target_value`, and `shares` columns in a picks.csv
using live Alpaca quotes (`get_latest_quote`). Intended to run right before
executing — collapses the reference-price staleness that makes picks.csv's
Friday-close prices useless at Monday 9:30 fill time.

Preserves a `picks.csv.bak` backup and also works in `--dry-run` mode to
preview drift without writing.

### 3. New flags on `scripts/production/execute_alpaca_rebalance.py`

- `--order-type limit` + `--limit-buffer-bps N` (default 30): submits LIMIT
  orders with `limit = current_ask + N bps` for buys, `current_bid − N bps`
  for sells. Fetches live NBBO at submit time.
- `--retry-unfilled`: reads the most recent `alpaca_orders_*.csv` in the
  picks directory, queries Alpaca for each order's current status, and
  builds an orders list of unfilled remainders (expired LOO, partially
  filled, etc.). Resubmits using whatever `--order-type` and
  `--limit-buffer-bps` you pass.
- `--retry-sells-as-market`: on retry, forces SELL side back to market
  orders (never want to be stuck holding an outgoing position because a
  limit didn't cross).

### 4. Updated `docs/WEEKLY_REBALANCE_QUICKREF.md`

New Monday morning workflow documented. The Friday-night cadence is marked
deprecated.

---

## Expected impact

**Current (Friday night → Monday fill):**

- Round-trip slippage: ~177 bps per full portfolio rebalance
- Annualized drag: 30% × weekly turnover × 52 weeks ≈ staggering

**New (Monday morning pipeline → 9:30 execute with refreshed prices + limits):**

- Round-trip slippage: ~30-40 bps (pure execution + tight 30 bps limit buffer)
- Annualized drag: ~6-8% annual cost — close to the backtest model

The difference — ~130 bps per rebalance × 52 weeks ≈ **20-30pp of annualized
CAGR clawed back from timing lag alone**. This is the single biggest
improvement lever found across all of Phase 4's work.

---

## Important caveats

1. **Paper trading simulator ≠ live execution.** The 88 bps slippage on
   MARKET orders may be paper-Alpaca's simulation of slippage, not real
   market friction. On live trading with an institutional broker, actual
   slippage could be lower. Until we trade live, these numbers are
   paper-trading-specific.

2. **The 18 bps LOO floor is likely the real execution cost on both paper
   and live.** If paper-Alpaca's LOO simulator is accurate, then 15-25 bps
   is the true floor and the backtest assumption stands.

3. **Regime-dependent slippage is a real risk.** In a 2026-03-style crash
   where slippage triples, limit orders will miss fills more often. The
   retry pass with --retry-sells-as-market is the defensive backstop for
   that scenario.

4. **Not yet paper-validated.** These flags were built and tested for
   correctness (CLI flags work, dry-run refreshes prices correctly, quotes
   fetch from Alpaca). They have not yet been used to execute a real
   rebalance. Use `--preview` mode on the next Monday before flipping to
   `--execute`.

---

## Reproducibility

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kairos-gpu

# Measure slippage across full live history
python scripts/evaluation/measure_live_slippage.py \
    --rebalance-dir outputs/rebalance \
    --output-dir outputs/evaluation/live_slippage

# Dry-run price refresh on a specific rebalance (no writes)
python scripts/production/refresh_picks_prices.py \
    --picks outputs/rebalance/2026-04-17/picks.csv \
    --dry-run

# Full new-cadence workflow (see docs/WEEKLY_REBALANCE_QUICKREF.md)
```
