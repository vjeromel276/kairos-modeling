# PRODUCTION POD V7 — FREEZE CONTRACT

Primary Engine:
- Script: backtest_academic_strategy_risk4_save.py
- Alpha: alpha_composite_v7
- Rebalance: 5 days
- Vol Target: 20%

Hedge Engine:
- Script: backtest_academic_strategy_ls_opt_save.py
- Alpha: alpha_composite_v33_regime
- Vol Target: 10%

Evaluation Window:
- Start: 2015-01-01
- End: rolling

Policy:
- No parameter changes
- No alpha changes
- No script changes without version bump (v7.1)

Research scripts remain present but must not be invoked.
