import pandas as pd
import matplotlib.pyplot as plt

# Load the P&L series (5-day avg return per rebalance)
pnl = pd.read_csv("mh_strategy_pnl.csv", index_col=0, parse_dates=True)["pnl"]

# Compute equity curve: product of (1 + return)
equity = (1 + pnl).cumprod()

# Plot
plt.figure(figsize=(10,6))
plt.plot(equity.index, equity.values) # type: ignore
plt.title("Equity Curve: Cumulative Return (Top 50, 5-day hold)")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.tight_layout()
plt.show()