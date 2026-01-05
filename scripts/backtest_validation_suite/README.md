# Backtest Validation Suite

Unified framework for backtesting and validating the Kairos ML alpha signal with configurable filters.

## Directory Structure

```
backtest_validation_suite/
├── configs/
│   ├── loose_2m.yaml      # $2M ADV, no price filter (current production)
│   ├── medium_10m.yaml    # $10M ADV, $5 price
│   └── tight_50m.yaml     # $50M ADV, $10 price (original validation)
├── results/               # Output directory (auto-created)
│   └── {timestamp}_{config_name}/
│       ├── config_used.yaml
│       ├── universe_stats.csv
│       ├── decile_analysis.csv
│       ├── decile_yearly.csv
│       ├── quintile_analysis.csv
│       ├── quintile_yearly.csv
│       ├── ic_monthly.csv
│       ├── ic_regime.csv
│       ├── backtest_returns.csv
│       └── summary.json
├── run_backtest_validation.py
├── compare_results.py
└── README.md
```

## Quick Start

### 1. Run a single configuration

```bash
# From kairos_phase4 directory
python scripts/backtest_validation_suite/run_backtest_validation.py \
  --config scripts/backtest_validation_suite/configs/tight_50m.yaml \
  --db data/kairos.duckdb \
  --output-dir scripts/backtest_validation_suite/results
```

### 2. Run all configurations

```bash
# Run all three filter configurations
for config in loose_2m medium_10m tight_50m; do
  python scripts/backtest_validation_suite/run_backtest_validation.py \
    --config scripts/backtest_validation_suite/configs/${config}.yaml \
    --db data/kairos.duckdb \
    --output-dir scripts/backtest_validation_suite/results
done
```

### 3. Compare results

```bash
# Compare all runs in results directory
python scripts/backtest_validation_suite/compare_results.py \
  --all scripts/backtest_validation_suite/results

# Or compare specific runs
python scripts/backtest_validation_suite/compare_results.py \
  results/20240105_123456_loose_2m \
  results/20240105_123789_tight_50m

# Save comparison to CSV
python scripts/backtest_validation_suite/compare_results.py \
  --all results/ \
  --output comparison.csv
```

## Configuration Options

### filters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `adv_min` | Minimum 20-day average daily volume ($) | 2,000,000 |
| `price_min` | Minimum stock price ($) | 0.0 |

### alpha
| Parameter | Description | Default |
|-----------|-------------|---------|
| `column` | Alpha signal column in feat_matrix_v2 | alpha_ml_v2_tuned_clf |
| `target` | Forward return column | ret_5d_f |

### portfolio
| Parameter | Description | Default |
|-----------|-------------|---------|
| `top_n` | Number of stocks to hold | 75 |
| `target_vol` | Target annual volatility | 0.25 |
| `max_position_pct` | Maximum weight per stock | 0.03 |
| `max_sector_mult` | Max sector weight as multiple of universe weight | 2.0 |

### turnover
| Parameter | Description | Default |
|-----------|-------------|---------|
| `lambda_tc` | Turnover smoothing (0=all new, 1=no change) | 0.20 |
| `turnover_cap` | Maximum turnover per rebalance | 0.20 |

### backtest
| Parameter | Description | Default |
|-----------|-------------|---------|
| `start_date` | Backtest start date | 2015-01-01 |
| `end_date` | Backtest end date | 2024-12-31 |
| `rebalance_every` | Days between rebalances | 5 |

## Output Files

### summary.json
Key metrics for comparison:
- Filter settings
- Universe size statistics
- Decile/quintile returns and spreads
- IC statistics
- Backtest performance (return, Sharpe, drawdown)

### decile_analysis.csv
10-bucket analysis showing:
- Annualized return per decile
- Sharpe ratio per decile
- Monotonicity check

### ic_monthly.csv
Monthly rolling Information Coefficient

### ic_regime.csv
IC broken down by market regime:
- UP vs DOWN markets
- HIGH_VOL vs LOW_VOL periods

### backtest_returns.csv
Period-by-period returns for the Risk4 portfolio

## Interpreting Results

### Good Signal Characteristics
- **Monotonic deciles**: D10 > D9 > ... > D1
- **Mean IC > 0.03**: Strong predictive power
- **IC % positive > 70%**: Consistent signal
- **Portfolio Sharpe > 1.5**: Strong risk-adjusted returns
- **Active Sharpe > 1.0**: Significant alpha over benchmark

### Filter Trade-offs
| Filter | Universe Size | Signal Quality | Practical Trading |
|--------|---------------|----------------|-------------------|
| Loose ($2M) | Large | May include noise | Harder to execute |
| Medium ($10M) | Moderate | Balanced | Reasonable |
| Tight ($50M) | Small | Cleanest | Easy execution |

## Updating Production Rebalancer

After finding optimal filters, update `generate_rebalance.py`:

```python
ML_CONFIG = {
    "min_adv": <optimal_adv>,      # From comparison
    "min_price": <optimal_price>,  # New field to add
    ...
}
```

And add price filter to `load_latest_data()` query.
