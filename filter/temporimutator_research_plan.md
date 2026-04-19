# Temporimutator Research Plan
## Kairos Phase 4 — Temporal Fusion Transformer for Equity Direction + Force
*Created: April 12, 2026*

---

## System Context

```
Project: Kairos Phase 4 — Temporimutator Track
Language: Python 3.11
Framework: PyTorch 2.6.0 + CUDA 12.4

Research DB (read-only, frozen snapshot):
  /data/kairos_research.duckdb  (inside container)

Production DB (read-only):
  /data/kairos.duckdb  (inside container)

Models out: /models/temporimutator/
Notebooks:  /workspace/

Hardware:
  GPU:  NVIDIA RTX 4060 Laptop, 8GB VRAM
  CPU:  8 physical / 16 logical cores
  RAM:  66.6 GB
  Disk: 1.1 TB free

MLflow: http://localhost:5000
```

---

## Purpose

Temporimutator is a separate research track from the Kairos XGBoost
cross-sectional model (v2_tuned / v3_neutral). It is not a replacement.
It learns latent causal structure from raw feature sequences — patterns
not expressible as point-in-time scalar features — and outputs two signals:

- **Direction** — up / flat / down over the next 5-20 trading days
- **Force** — expected magnitude of move, scaled by volatility

The output of Temporimutator becomes an additional signal in the Kairos
ensemble alongside alpha_ml_v3_neutral_clf.

---

## Architecture Overview

### Input Representation

For each stock on each inference date, the model receives a 252-trading-day
lookback window (approximately 1 calendar year) represented as 4 feature
streams plus 1 volatility scalar.

**Four input streams (each z-scored within the window):**

| Stream | Formula | What it captures |
|--------|---------|-----------------|
| RSI(14) | Standard Wilder RSI from closeadj | Cycle position — overbought/oversold rhythm |
| Volume ratio | volume / rolling_20d_avg(volume) | Participation — conviction behind moves |
| Trend extension | closeadj / EMA(50) - 1 | Distance from trend — extension and exhaustion |
| ATR ratio | ATR(14) / closeadj | Volatility regime — expansion and contraction |

ATR(14) = Wilder smoothed average of:
  max(high-low, |high-prev_close|, |low-prev_close|) over 14 days

**Volatility scalar (not z-scored — appended separately):**

  vol_scalar = std(daily log returns of closeadj over the 252-day window)

This scalar is NOT fed into the transformer. It is concatenated to the
transformer output before the prediction heads. It gives the model back
the absolute volatility information that z-scoring removed, so the force
head can predict meaningful magnitudes.

**Input tensor shape per sample:**
  sequence:      (252, 4)   — 252 days × 4 streams
  vol_scalar:    (1,)       — single float

### Transformer Architecture

```
Input: (batch, 252, 4)
  ↓
Linear projection: (batch, 252, d_model=64)
  ↓
Positional encoding (learnable, not sinusoidal)
  ↓
Transformer encoder:
  - 3 layers
  - 4 attention heads
  - d_model = 64
  - d_ff = 256 (feedforward hidden dim)
  - dropout = 0.1
  ↓
Global average pooling over sequence dim: (batch, 64)
  ↓
Concatenate vol_scalar: (batch, 65)
  ↓
┌─────────────────────────┐    ┌──────────────────────────┐
│  Direction head          │    │  Force head               │
│  Linear(65, 32) → ReLU  │    │  Linear(65, 32) → ReLU   │
│  Linear(32, 3)           │    │  Linear(32, 1)            │
│  Softmax                 │    │  (raw — no activation)    │
│  [down, flat, up]        │    │  expected |return|        │
└─────────────────────────┘    └──────────────────────────┘
```

**Why learnable positional encoding:**
Financial sequences have no fixed periodicity. Sinusoidal encoding assumes
uniform spacing and fixed patterns. Learnable encoding lets the model
discover which positions in the year carry the most predictive information.

**Why global average pooling:**
Attention-weighted pooling is an alternative but average pooling forces
the model to build representations in the attention layers rather than
relying on a pooling gate. Simpler and more robust at this data scale.

**VRAM estimate:**
  Model parameters: ~500K (well within 8GB)
  Batch size 512:   512 × 252 × 4 × 4 bytes ≈ 2MB per batch
  Peak VRAM:        ~3-4GB training, leaving headroom for optimizer states

---

## Data Pipeline

### Source Table
`sep_base_academic` in research DB:
  - ticker, date, open, high, low, close, volume, closeadj
  - 2,335 tickers, 2015-01-02 to 2025-12-31, 5,288,553 rows
  - Always use closeadj for price calculations

### Feature Computation (per ticker, chronological)

```python
# RSI(14) — Wilder smoothing
delta = closeadj.diff()
gain  = delta.clip(lower=0)
loss  = (-delta).clip(lower=0)
avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
rs  = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Volume ratio
vol_ratio = volume / volume.rolling(20).mean()

# Trend extension
ema50 = closeadj.ewm(span=50, adjust=False).mean()
trend_ext = closeadj / ema50 - 1

# ATR(14) — Wilder smoothing
hl  = high - low
hpc = (high - closeadj.shift(1)).abs()
lpc = (low  - closeadj.shift(1)).abs()
tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
atr_ratio = atr / closeadj
```

### Window Construction

- Window length: 252 trading days
- Step size:     21 trading days
- Min history required: 252 + 20 days (window + max label horizon)
- Label: sign and magnitude of closeadj return over next N days
  where N ∈ {5, 10, 20}

**Per window, compute:**
1. Extract 252-day sequence of [rsi, vol_ratio, trend_ext, atr_ratio]
2. Z-score each stream independently within the window:
   stream_z = (stream - stream.mean()) / (stream.std() + 1e-8)
3. Compute vol_scalar = std(log(closeadj / closeadj.shift(1))) over window
4. Compute label:
   ret_Nd = closeadj.iloc[window_end + N] / closeadj.iloc[window_end] - 1
   direction = 1 if ret_Nd > threshold else (-1 if ret_Nd < -threshold else 0)
   force = abs(ret_Nd) / vol_scalar  # vol-normalized magnitude
5. Drop window if any stream has >10% missing values

**Direction threshold:**
Use the cross-sectional median absolute return for that label horizon
as the flat zone boundary. Stocks within ±median/2 are labeled flat.
This produces approximately equal class balance across up/flat/down.

### Approximate Dataset Size

| Split | Window start dates | Approx samples |
|-------|--------------------|----------------|
| Train | < 2022-03-01 | ~200,000 |
| Val   | 2023-01-01 → 2023-12-31 | ~28,000 |
| Test  | 2024-01-01 → 2024-12-31 | ~28,000 |

Note: Gap between train end and val start explained below.

---

## Train / Validation / Test Split — Critical Design

This is the most important correctness guarantee in the entire pipeline.
Rolling windows create severe contamination risk if splits are done naively.

### The contamination problem

A training window starting 2022-10-01 ends 2023-09-30 (252 trading days).
Its label uses closeadj on day 2023-10-20 (20 days forward).
A validation window starting 2023-01-01 uses closeadj from 2023-01-01
onward as features.

These two windows share overlapping feature data (Jan-Sep 2023).
If the first is in training and the second in validation, the model
has seen the future during training. This is direct data leakage.

### The fix — purge gap scaled to window length

```
PURGE GAP = window_length + max_label_horizon = 252 + 20 = 272 trading days

Validation start:  2023-01-01
Purge boundary:    2023-01-01 minus 272 trading days ≈ 2022-03-01

Training windows:   start_date < 2022-03-01  (end+label clears val start)
Validation windows: start_date 2023-01-01 → 2023-12-31
Test windows:       start_date 2024-01-01 → 2024-12-31

Gap between train and val: ~10 months of unused data
This is the price of correctness. Accept it.
```

### Why not CPCV here

CPCV with C(6,2) combinations works well for point-in-time features
where each row is independent. For 252-day overlapping windows, CPCV
combinatorics produce many fold boundaries each requiring a 272-day
purge gap. The usable training data per fold drops below reliable
training thresholds. Chronological train/val/test with a single large
purge gap is the correct approach for long-window sequence models.

### Implementation rule

Always filter windows by `window_start_date`, never by row index.
Store window_start_date, window_end_date, and label_date explicitly
in the dataset so leakage can be audited at any time.

---

## Label Design

Three label horizons trained as separate model variants:

| Variant | Horizon | Use case |
|---------|---------|----------|
| TM-5 | 5 trading days | Weekly rebalance alignment with v3_neutral |
| TM-10 | 10 trading days | Bi-weekly |
| TM-20 | 20 trading days | Monthly |

Start with TM-5 to allow direct IC comparison with v3_neutral (same horizon).

**Direction labels (3-class):**
  up:   ret_5d > +threshold
  flat: -threshold <= ret_5d <= +threshold
  down: ret_5d < -threshold

**Force labels (regression):**
  force = abs(ret_5d) / vol_scalar
  Winsorize at 99th percentile to remove outlier events

**IC evaluation:**
Always compute IC against raw ret_5d_f using the direction probability
(P(up) - P(down)) as the ranking signal. This keeps evaluation
methodology identical to v3_neutral for direct comparison.

---

## MLflow Experiment Structure

| Experiment | Purpose |
|---|---|
| `temporimutator_data_pipeline` | Validate dataset construction, confirm no leakage |
| `temporimutator_architecture` | Forward pass tests, VRAM profiling, ablations |
| `temporimutator_training` | Full training runs, walk-forward evaluation |
| `temporimutator_inference` | Live universe scoring, direction + force output |

**Standard metrics to log for every training run:**
  - val_direction_accuracy (3-class)
  - val_direction_ic (Spearman of P(up)-P(down) vs ret_5d_f)
  - val_force_mae
  - test_direction_ic
  - attention_entropy (mean entropy of attention weights — measures focus)
  - n_train_windows, n_val_windows, n_test_windows

**Baseline to beat:**
  CPCV IC 0.0259 / IC Sharpe 3.727 (from v3_neutral / v2_raw)
  Any Temporimutator variant must exceed this on the test set
  before being considered for ensemble inclusion.

---

## Notebook Plan

### Notebook 1 — `temporimutator_data_pipeline.ipynb`

1. Compute all 4 streams for full universe from research DB
2. Construct rolling windows with 21-day step
3. Z-score each stream within window
4. Compute vol_scalar per window
5. Compute direction and force labels for 5d, 10d, 20d
6. Apply train/val/test split with 272-day purge gap
7. **Leakage audit:** confirm zero window_start_date overlap across splits
8. Save dataset to /models/temporimutator/ as compressed numpy arrays
9. Log dataset statistics to MLflow temporimutator_data_pipeline

**Outputs:**
  /models/temporimutator/train_sequences.npy   — shape (N_train, 252, 4)
  /models/temporimutator/train_scalars.npy     — shape (N_train, 1)
  /models/temporimutator/train_labels_dir.npy  — shape (N_train,) int [0,1,2]
  /models/temporimutator/train_labels_force.npy — shape (N_train,) float
  /models/temporimutator/train_meta.parquet    — ticker, window_start, window_end, label_date
  (same pattern for val_ and test_)

### Notebook 2 — `temporimutator_architecture.ipynb`

1. Define TemporimutatorModel in PyTorch
2. Forward pass test on random data — confirm output shapes
3. VRAM profiling at batch sizes 128, 256, 512
4. Attention weight visualization on 3 sample stocks
5. Parameter count and gradient flow check
6. Log architecture config to MLflow temporimutator_architecture

**Outputs:**
  /models/temporimutator/model_config.json  — architecture hyperparameters

### Notebook 3 — `temporimutator_training.ipynb`

1. Load train/val datasets
2. Train TM-5 (5-day horizon) first
3. AdamW optimizer, cosine LR schedule with warmup
4. Early stopping on val_direction_ic (not loss)
5. Log per-epoch metrics to MLflow
6. Save best checkpoint by val IC
7. Evaluate on test set — compute IC, compare to CPCV baseline
8. Attention weight analysis — which days and which streams matter most
9. Repeat for TM-10 and TM-20

**Outputs:**
  /models/temporimutator/tm5_best.pt
  /models/temporimutator/tm10_best.pt
  /models/temporimutator/tm20_best.pt

### Notebook 4 — `temporimutator_inference.ipynb`

1. Load production DB
2. Compute 4 streams for live cross-section (most recent 252 days)
3. Z-score within window, compute vol_scalar
4. Load best TM-5 checkpoint
5. Generate direction probabilities and force scores for all 2,294 stocks
6. Compute composite signal: (P(up) - P(down)) * force
7. Write to predictions DB alongside v3_neutral predictions
8. Log to MLflow temporimutator_inference

**Output table in /models/kairos_predictions.duckdb:**
  temporimutator_v1  — ticker, date, p_up, p_flat, p_down, force, tm_signal

---

## Feature Combination Sweep — Future Path

Once the base 4-stream model is proven, the combinatorial sweep:

1. Generate all C(25, 2) = 300 pairs from full Kairos feature set
2. For each pair, train a 2-stream Temporimutator with identical architecture
3. Evaluate CPCV IC for each pair
4. Rank pairs by IC Sharpe
5. Top 10 pairs → train 4-stream combinations from best pairs
6. The attention weights from the winning combinations reveal which
   feature interactions are genuinely causal

This runs as MLflow experiment `temporimutator_feature_sweep`.
Estimated runtime: 300 pairs × ~20 min each = ~100 GPU hours.
Run overnight in batches of 50.

---

## Key Decisions

1. Use closeadj for all price calculations — handles splits and dividends.
2. Z-score each stream within window — removes level and volatility bias.
3. Vol_scalar appended after transformer — gives force head scale information.
4. Learnable positional encoding — financial sequences have no fixed periodicity.
5. 272-day purge gap between train and val — non-negotiable correctness requirement.
6. Evaluate IC against raw ret_5d_f — identical methodology to v3_neutral.
7. Baseline to beat: CPCV IC 0.0259, IC Sharpe 3.727.
8. TM-5 first — aligns with weekly rebalance and allows direct comparison.
9. Temporimutator output feeds ensemble alongside v3_neutral, not as replacement.
10. All outputs written to /models/temporimutator/ and logged to MLflow.

---

## Naming Convention

| Component | Name |
|---|---|
| Track name | Temporimutator |
| Model files | tm5_best.pt, tm10_best.pt, tm20_best.pt |
| Predictions table | temporimutator_v1 |
| MLflow experiments | temporimutator_* |
| Signal column | tm_signal (composite direction × force) |

---

*Document maintained across chat sessions.*
*Always include kairos_ml_research_status alongside this document in every new chat.*
