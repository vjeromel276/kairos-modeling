# Rebalance Progress - January 12, 2026

## Status: ACTION REQUIRED

The January 9 rebalance **failed to execute properly**. 46 out of 55 sell orders expired without filling, leaving the portfolio misaligned.

---

## Root Cause Analysis

### What Happened

| Metric | Count |
|--------|-------|
| Total sell orders submitted | 55 |
| Orders filled | 9 |
| Orders expired | 46 |
| Fill rate | 16.4% |

### Timeline

- **Saturday Jan 10, 12:38 PM**: Rebalance script executed, orders submitted to Alpaca
- **Monday Jan 12, 04:02 AM ET**: Orders queued for market open (per Alpaca API timestamps)
- **Monday Jan 12, 09:30-09:32 AM ET**: 46 orders expired at market open instead of filling

### Root Cause: Alpaca Paper Trading MOO Limitation

**Alpaca's paper trading environment does not properly simulate Market-on-Open (MOO) orders.**

From the [Alpaca Community Forum](https://forum.alpaca.markets/t/accurate-opg-and-cls-prices-for-paper-trading/3762):

> "Paper trading treats OPG orders as regular market orders that fill at the current bid/ask spread, rather than simulating actual opening auction mechanics."

> "Approximately 26.7% of OPG orders experienced problems, with some orders being canceled roughly one minute after market open without execution."

> "Alpaca confirmed this limitation exists and stated improved MOO/MOC simulation 'is on our roadmap,' though no timeline was provided."

### Why Some Orders Filled

The 9 orders that filled (RZLT, COLL, MCD, DOMO, HOG, AUPH, CVI, GIII, MSCI) likely executed due to favorable timing in the paper trading simulation. There's no clear pattern - it appears random.

---

## Current Portfolio State

| Metric | Value |
|--------|-------|
| Current positions | 84 |
| Target positions | 75 |
| Positions to SELL | 46 |
| Positions to BUY | 37 |
| Account equity | $101,402.88 |
| Cash available | $5,530.02 |
| Buying power | $104,913.40 |

---

## Action Required: Monday January 13, 2026

### Recommended Approach

Use **intraday market orders** instead of MOO orders to avoid the paper trading limitation:

```bash
cd /media/vjl2dev/b1eb2f9b-513e-4494-a9fa-9c137dd6f81b/media/vjerome2/Extreme\ Pro/kairos_phase4

# Activate environment
conda activate kairos-gpu

# Execute rebalance with intraday market orders
python scripts/production/execute_alpaca_rebalance.py \
  --picks outputs/rebalance/2026-01-09/picks.csv \
  --execute --intraday
```

The `--intraday` flag uses immediate market orders (`time_in_force=day`) instead of MOO orders (`time_in_force=opg`).

---

## Positions to SELL (46 stocks, ~$49,695)

These positions are held but NOT in the target picks:

| Symbol | Qty | Price | Value | Action |
|--------|-----|-------|-------|--------|
| AIRO | 66 | $12.70 | $838.20 | SELL ALL |
| ALTO | 239 | $2.53 | $604.67 | SELL ALL |
| ANAB | 17 | $49.27 | $837.59 | SELL ALL |
| AVDL | 92 | $21.45 | $1,973.40 | SELL ALL |
| AXGN | 19 | $31.62 | $600.82 | SELL ALL |
| AXSM | 3 | $173.20 | $519.60 | SELL ALL |
| BRBR | 28 | $22.62 | $633.36 | SELL ALL |
| BTBT | 195 | $2.28 | $444.60 | SELL ALL |
| CAH | 8 | $208.51 | $1,668.08 | SELL ALL |
| CME | 10 | $264.83 | $2,648.30 | SELL ALL |
| CNA | 51 | $46.40 | $2,366.40 | SELL ALL |
| COR | 7 | $344.00 | $2,408.00 | SELL ALL |
| CORT | 10 | $37.20 | $372.00 | SELL ALL |
| CPK | 10 | $123.49 | $1,234.90 | SELL ALL |
| CR | 9 | $202.93 | $1,826.37 | SELL ALL |
| CSTL | 22 | $39.66 | $872.52 | SELL ALL |
| DBI | 48 | $7.92 | $380.16 | SELL ALL |
| DFH | 58 | $19.91 | $1,154.78 | SELL ALL |
| EA | 17 | $203.81 | $3,464.77 | SELL ALL |
| ESTA | 13 | $66.85 | $869.05 | SELL ALL |
| FERG | 6 | $241.53 | $1,449.18 | SELL ALL |
| FTRE | 43 | $18.39 | $790.77 | SELL ALL |
| GWW | 1 | $1,032.31 | $1,032.31 | SELL ALL |
| HAS | 20 | $86.60 | $1,732.00 | SELL ALL |
| HCSG | 6 | $19.43 | $116.58 | SELL ALL |
| IDXX | 1 | $724.76 | $724.76 | SELL ALL |
| ISSC | 24 | $19.76 | $474.24 | SELL ALL |
| JCI | 11 | $111.32 | $1,224.52 | SELL ALL |
| LC | 11 | $20.09 | $220.99 | SELL ALL |
| LENZ | 27 | $15.55 | $419.85 | SELL ALL |
| LRN | 11 | $68.98 | $758.78 | SELL ALL |
| LUMN | 69 | $7.96 | $549.24 | SELL ALL |
| LW | 17 | $41.43 | $704.31 | SELL ALL |
| MAMA | 31 | $12.97 | $402.07 | SELL ALL |
| MCK | 1 | $825.61 | $825.61 | SELL ALL |
| MDB | 2 | $419.68 | $839.36 | SELL ALL |
| MNST | 21 | $77.48 | $1,627.08 | SELL ALL |
| MO | 2 | $58.33 | $116.66 | SELL ALL |
| NTRA | 6 | $235.33 | $1,411.98 | SELL ALL |
| NUTX | 3 | $179.00 | $537.00 | SELL ALL |
| ODC | 14 | $51.77 | $724.78 | SELL ALL |
| ORI | 54 | $42.52 | $2,296.08 | SELL ALL |
| PSIX | 7 | $71.00 | $497.00 | SELL ALL |
| SAFT | 23 | $76.49 | $1,759.27 | SELL ALL |
| STE | 8 | $260.77 | $2,086.16 | SELL ALL |
| TDAY | 110 | $5.97 | $656.70 | SELL ALL |

**Total Sell Value: ~$49,695**

---

## Positions to BUY (37 stocks, ~$31,838)

These are in the target picks but NOT currently held:

| Symbol | Shares | Price | Target Value | Weight |
|--------|--------|-------|--------------|--------|
| PLNT | 8 | $106.85 | $938.06 | 0.93% |
| PLTK | 223 | $3.62 | $809.45 | 0.80% |
| PPG | 11 | $107.41 | $1,182.35 | 1.17% |
| RL | 2 | $369.81 | $994.74 | 0.98% |
| RLI | 23 | $60.15 | $1,387.95 | 1.37% |
| RPM | 9 | $111.02 | $1,078.50 | 1.07% |
| RRR | 11 | $62.65 | $707.01 | 0.70% |
| RXST | 62 | $8.84 | $553.54 | 0.55% |
| RYM | 9 | $20.01 | $186.50 | 0.18% |
| RYTM | 4 | $101.02 | $443.93 | 0.44% |
| SEPN | 23 | $25.30 | $594.36 | 0.59% |
| SGI | 9 | $93.78 | $888.56 | 0.88% |
| SM | 54 | $17.78 | $974.64 | 0.96% |
| SMCI | 15 | $30.16 | $479.99 | 0.48% |
| SNA | 3 | $362.17 | $1,327.43 | 1.31% |
| STLD | 4 | $169.27 | $832.99 | 0.82% |
| STWD | 77 | $18.23 | $1,408.96 | 1.39% |
| TARS | 11 | $76.87 | $890.88 | 0.88% |
| TDG | 1 | $1,392.09 | $1,509.69 | 1.49% |
| TLN | 0 | $370.83 | $166.60 | 0.16% |
| TNGX | 34 | $11.75 | $404.32 | 0.40% |
| TPR | 6 | $134.35 | $849.66 | 0.84% |
| TT | 2 | $381.70 | $990.90 | 0.98% |
| TTEK | 18 | $36.33 | $662.96 | 0.66% |
| TVTX | 11 | $34.58 | $395.84 | 0.39% |
| TXNM | 15 | $58.95 | $908.56 | 0.90% |
| TXO | 94 | $10.50 | $989.18 | 0.98% |
| UAN | 8 | $109.10 | $962.92 | 0.95% |
| UHS | 5 | $207.03 | $1,135.27 | 1.12% |
| VEL | 49 | $18.62 | $930.08 | 0.92% |
| VMI | 2 | $425.58 | $1,110.72 | 1.10% |
| WM | 8 | $220.91 | $1,796.72 | 1.78% |
| WVE | 8 | $13.84 | $117.70 | 0.12% |
| XNCR | 29 | $13.89 | $409.07 | 0.40% |
| XYL | 13 | $139.69 | $1,875.12 | 1.86% |
| ZIP | 170 | $3.26 | $554.25 | 0.55% |
| ZYME | 16 | $23.17 | $388.66 | 0.38% |

**Total Buy Value: ~$31,838**

---

## Financial Summary

| Item | Amount |
|------|--------|
| Sell proceeds (estimated) | $49,695 |
| Buy cost (estimated) | $31,838 |
| **Net cash release** | **$17,857** |
| Current cash | $5,530 |
| Post-rebalance cash (est.) | $23,387 |

The sells exceed buys, so there's no buying power issue.

---

## Long-Term Recommendations

### 1. Use Intraday Orders for Paper Trading

For paper trading, always use `--intraday` flag to avoid MOO order issues:

```bash
python scripts/production/execute_alpaca_rebalance.py --picks <picks.csv> --execute --intraday
```

### 2. Consider Limit Orders (Future Enhancement)

To avoid adverse price movements, consider adding limit order support with a small buffer (e.g., 0.25% above ask for buys, 0.25% below bid for sells).

### 3. For Live Trading

MOO orders work properly in live trading. The issue is specific to Alpaca's paper trading simulation. When moving to live trading, MOO orders should be reliable.

---

## Quick Reference Commands

```bash
# Activate environment
conda activate kairos-gpu

# Preview what will happen (no execution)
# Note: Paper trading auto-detects and uses intraday orders
python scripts/production/execute_alpaca_rebalance.py \
  --picks outputs/rebalance/2026-01-09/picks.csv --preview

# Execute the rebalance (during market hours)
# Paper trading will automatically use intraday market orders
python scripts/production/execute_alpaca_rebalance.py \
  --picks outputs/rebalance/2026-01-09/picks.csv --execute

# Force MOO orders on paper trading (for testing only)
python scripts/production/execute_alpaca_rebalance.py \
  --picks outputs/rebalance/2026-01-09/picks.csv --execute --force-moo

# Check current holdings
python scripts/alpaca.py
```

---

## Script Update (January 12, 2026)

The `execute_alpaca_rebalance.py` script has been updated to **automatically use intraday orders for paper trading**.

- Paper trading is detected via the API URL (contains "paper")
- No need to pass `--intraday` flag anymore for paper trading
- Use `--force-moo` if you want to test MOO behavior on paper (not recommended)
- Live trading will still default to MOO orders as intended

---

*Document generated: January 12, 2026*
*Updated: January 12, 2026 - Added auto-intraday for paper trading*
