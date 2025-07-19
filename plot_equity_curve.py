import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the P&L series (5-day avg **log** return per rebalance)
df = pd.read_csv("mh_strategy_pnl.csv", index_col=0, parse_dates=True)
log_rets = df["pnl"]

# 1) Cumulative log‐return equity curve
equity_log = np.exp(log_rets.cumsum())

# 2) Convert to simple returns and build that equity curve
simple_rets = np.exp(log_rets) - 1
equity_simple = (1 + simple_rets).cumprod()

# Plot both
plt.figure(figsize=(10, 6))
plt.plot(equity_log.index, equity_log.values, label="Log‐return curve") # type: ignore
plt.plot(equity_simple.index, equity_simple.values, label="Simple‐return curve", alpha=0.8) # type: ignore
plt.title("Equity Curves: Log vs Simple Cumulative Return\n(Top 50, 5-day hold)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
