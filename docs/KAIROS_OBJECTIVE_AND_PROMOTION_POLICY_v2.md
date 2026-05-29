# Kairos Objective Hierarchy and Promotion Policy

**Version:** 2.0  
**Last Updated:** 2026-04-16  
**Status:** Active — governs all research, strategy, and deployment decisions

---

## Table of Contents

1. [Project Creed](#1-project-creed)
2. [Objective Hierarchy](#2-objective-hierarchy)
3. [Layer Objectives and Metrics](#3-layer-objectives-and-metrics)
4. [Promotion Gates](#4-promotion-gates)
5. [Implementation Phases](#5-implementation-phases)
6. [Concrete Roadmap](#6-concrete-roadmap)
7. [Key Principles](#7-key-principles)

---

## 1. Project Creed

> Kairos exists to generate repeatable, risk-adjusted returns by deploying only those signals and exposure rules that remain robust across market regimes, with signal-quality metrics serving as research tools rather than final objectives.

Accuracy is a useful priest. IC is a useful priest. Profit is a loud deacon with bad impulse control. But the god of the system should be **durable compounding under uncertainty**.

---

## 2. Objective Hierarchy

The project's ordered objective, non-negotiable:

### Priority 1: Survive

No catastrophic drawdowns, no regime-specific blowups, no broken short book quietly lighting the carpet on fire. The repo's own regime analysis says low-vol regimes are hurting performance, the short signal is fundamentally broken, and long-only currently beats the regime-switcher.

### Priority 2: Compound

Maximize risk-adjusted return, not hit rate. The current tools already use Sharpe, max drawdown, annualized return, active return, and volatility targeting — exactly the right family of metrics for this level.

### Priority 3: Select Signal Intelligently

Use IC, CPCV IC Sharpe, and fold consistency to decide which models deserve promotion into the capital-allocation layer. The research plan explicitly says CPCV is the standard methodology, IC Sharpe is the model-selection criterion, and IC is always evaluated against raw `ret_5d_f`.

### What This Means the Project Should NOT Treat as Top Objective

- **Accuracy** — useful diagnostic, not the goal
- **Profit alone** — ignores the path taken to get there
- **Minimum loss** — optimizes for timidity, not compounding

Those are all false gods. Very respectable false gods, wearing ties and carrying spreadsheets, but false gods nonetheless.

---

## 3. Layer Objectives and Metrics

| Layer | Objective | Primary Metrics |
|-------|-----------|-----------------|
| **Research** | Maximize out-of-sample signal quality and robustness | CPCV IC Sharpe, mean IC, fold consistency |
| **Strategy** | Maximize risk-adjusted return across regimes | Sharpe, annualized return, max drawdown, regime-specific stability |
| **Deployment** | Prefer strategies that remain acceptable across bull/bear and high/normal/low vol environments | Regime-conditioned performance, transition robustness |

The hierarchy is explicit because the pieces exist but the contracts are still partly implicit. Future experiments must not wander off worshipping a sexy metric.

---

## 4. Promotion Gates

### 4.1 Research Promotion (Model → Strategy Evaluation)

A model may advance to strategy-level evaluation only if it meets **all** of:

| Criterion | Description |
|-----------|-------------|
| CPCV mean IC | Above threshold |
| CPCV IC Sharpe | Above threshold |
| Positive folds | High fraction of positive folds |
| Data integrity | No data-join contamination |
| Forward-return validity | No invalid `ret_5d_f` period beyond research DB clean boundary (currently 2025-12-12) |

Reference: The proven v3 neutral model demonstrates this gate — CPCV mean IC 0.0259, IC Sharpe 3.727, 100% positive folds.

IC and CPCV belong in the model-research layer. They must not act as the final arbiter of live desirability. IC finds signal; the backtester decides whether the signal deserves money.

### 4.2 Strategy Promotion (Strategy → Deployment)

A strategy may be deployed only if it meets **all** of:

| Criterion | Description |
|-----------|-------------|
| Sharpe | Above threshold |
| Max drawdown | Below threshold |
| Regime performance | Acceptable in all major regimes, or explicit gating to avoid bad ones |
| Regime concentration | No single regime responsible for nearly all profits (unless intentionally designed that way) |
| Production path | Live-realizable workflow through production DB and prediction script path |

Research heroics that cannot survive the production plumbing are just expensive fan fiction.

Strategy-level metrics come from portfolio backtests and regime-conditioned analysis: Sharpe, annual return, max drawdown, performance by bull/bear, performance by vol regime, and transition robustness around regime changes. These are already present in `deep_regime_analysis.py` and the optimized LS backtester.

---

## 5. Implementation Phases

### Phase 1 — Formalize the Objective in Writing

Create this document. Define the Kairos objective function and how every layer is judged, making the contracts explicit.

### Phase 2 — Separate Research Metrics from Capital Metrics

Keep IC and CPCV in the model-research layer. Do not let them act as the final arbiter of live desirability.

- **Model promotion gate:** CPCV IC Sharpe, mean IC, percentage of positive folds
- **Strategy promotion gate:** Portfolio backtests and regime-conditioned metrics (Sharpe, annual return, max drawdown, bull/bear performance, vol regime performance, transition robustness)

This preserves the existing contract: IC finds signal; the backtester decides whether the signal deserves money.

### Phase 3 — Add an EV/Regime Evaluation Layer for Predictions

The missing bridge between model output and portfolio outcome. Build a new evaluation module that takes predictions plus realized `ret_5d_f` and reports:

- Prediction bucket / decile
- Count, win rate, average return, median return
- Average win, average loss, expected value per bucket
- Cumulative return by bucket, drawdown by bucket
- All metrics split by regime

This layer answers: "Where does the money come from?", "Which confidence bands are actually worth trading?", and "Does the model keep positive EV in low-vol, high-vol, bull, and bear conditions?"

This transforms the project from "predictive modeling with nice charts" into "signal triage under changing market weather."

### Phase 4 — Make Regime Robustness a First-Class Promotion Criterion

Promote the existing regime framework from analysis script to policy. Hard rules:

- No strategy promoted if one regime shows catastrophic max drawdown or materially negative Sharpe, unless an explicit allocation rule turns the strategy off there
- A strategy strong only in high-vol regimes may be promoted, but only behind a regime gate or exposure throttle
- If long-only dominates across most regimes and L/S helps only marginally in bear markets, L/S should not be the default capital path until the short book is fixed

The idol is not "always be invested the same way." The idol is "compound with discipline under state changes."

### Phase 5 — Implement a Two-Stage Decision System

Do not make the model decide everything.

**Stage A — Signal:** Models output predicted return or ranking signal. This is the training and inference territory. The immediate next step is fixing `generate_ml_predictions_v3_neutral.py` to use the proven neutral model and live features.

**Stage B — Portfolio / Exposure:** A downstream controller decides:

- Whether signal strength is high enough
- Whether the current regime allows full exposure
- Whether to go long-only, reduced risk, cash, or a different sleeve
- Whether the short book is even allowed

This stage consumes: signal quality, regime classification, drawdown state, turnover/friction constraints, risk limits. It matches the backtest code's existing orientation: mean-variance optimization, sector caps, stock caps, shrinkage, volatility targeting.

### Phase 6 — Fix the Short Book Only After the Objective Layer Is in Place

The regime analysis is clear: the short signal is broken, bottom decile outperforms, and L/S only helps marginally in bear markets.

- Do not make L/S central right now
- Treat long-only plus regime-aware exposure as the current production spine
- Research a separate short signal (distinct features and evaluation rules) instead of assuming the long alpha inverts cleanly
- Only short in bear/high-vol regimes

Many systems die trying to force symmetry where the market provides none. The market is under no moral obligation to let your bottom decile behave like an elegant mirror image.

### Phase 7 — Promote Vol Timing Before Cleverness

The simplest likely improvement already exists in the analysis:

- High-vol regimes are the sweet spot
- Low-vol regimes are weak
- High-vol / high+normal-vol gating is explicitly tested
- The analysis recommends implementing the simplest version first

Implementation order:

1. Prediction EV by bucket and regime
2. Long-only exposure gating by vol regime
3. Drawdown-aware exposure scaling
4. Only then more complex multi-strategy sleeves

That is how you keep contracts and avoid "Look, Ma, I invented ten interacting optimizers and now none of them can be trusted."

### Phase 8 — Define Promotion Criteria That Match the Objective

Codify the gates from Section 4 into enforceable policy. Every model and strategy must clear both the research and strategy gates before deployment. No exceptions.

---

## 6. Concrete Roadmap

Definitive execution order:

| Step | Action | Dependencies |
|------|--------|-------------|
| 1 | Document the objective hierarchy: survive → compound → select signal | None (this document) |
| 2 | Fix prediction script for proven neutral model — stable live predictions from the right model artifact and feature list | Step 1 |
| 3 | Add prediction evaluation module: bucket predictions, compute EV stats, compute bucket drawdowns, split by regime | Step 2 |
| 4 | Integrate regime tags into evaluation using existing `regime_history` logic and bull/bear + vol-regime concepts from `deep_regime_analysis.py` | Step 3 |
| 5 | Define regime-aware exposure rules for long-only sleeve: full / reduced / zero exposure by regime; test simplest vol-gating first | Step 4 |
| 6 | Backtest exposure rules with existing portfolio framework; compare: current long-only, current regime switch, long-only + vol gating, long-only + drawdown throttle | Step 5 |
| 7 | Research separate short book: distinct target or feature family, separate evaluation, only active in allowed regimes | Step 6 |
| 8 | Promote only strategies that improve the actual objective: better Sharpe, acceptable drawdown, stable regime performance, clean implementation path | Ongoing |

---

## 7. Key Principles

**On regime gating:** No strategy can be promoted if one regime shows catastrophic max drawdown or materially negative Sharpe unless there is an explicit allocation rule that turns the strategy off in that regime.

**On the short book:** The short signal is currently broken. Long-only plus regime-aware exposure is the production spine until a separate short signal is researched and validated independently.

**On L/S vs long-only:** If long-only dominates across most regimes and L/S helps only marginally in bear markets, then L/S should not be the default capital path until the short book is fixed.

**On vol timing:** The simplest likely improvement — vol-regime gating — should be implemented before any complex multi-strategy sleeves.

**On metric hierarchy:** Research metrics (IC, CPCV) are selection tools, not final objectives. Portfolio metrics (Sharpe, drawdown, regime stability) are the arbiters of deployment worthiness.

**On production realizability:** A strategy that cannot survive the production plumbing (DB path, prediction scripts, rebalancer, execution) does not get promoted regardless of research performance.
