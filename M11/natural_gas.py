"""
Natrual gas arbitrage

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 0. Data Management
# ------------------------------------------------------------
#Load data
data = pd.read_csv("Commodity.csv")

#Year and month
data["Year"] = data["Time"].str[:4].astype(int)
data["Month"] = data["Time"].str[-2:].astype(int)

#Visualisation: natural gas traded in two locations
time = data["Year"] + (data["Month"] - 1) / 12

plt.plot(time, data["Natural gas, Europe"], label="Europe")
plt.plot(time, data["Natural gas, US"], label="US")
plt.xlabel("Year")
plt.ylabel("Natural Gas")
plt.legend()
plt.tight_layout()

#Save figure
plt.savefig("Figure1.png")
plt.show()

#Ratio analysis
data["US_EU"] =  data["Natural gas, US"]/data["Natural gas, Europe"]
plt.plot(data["Year"] + (data["Month"] - 1) / 12, data["US_EU"])
plt.xlabel("Year")
plt.ylabel("US-Europe Ratio of Natural Gas Prices")
plt.tight_layout()

#Save figure
plt.savefig("Figure2.png")
plt.show()


# ------------------------------------------------------------
# 1. Parameters & Assumptions
# ------------------------------------------------------------
LOOKBACK = 36         # Historical window in months
ENTRY_Z = 1.5         # Enter when ratio is unusually high/low
EXIT_Z = 0.25         # Exit when ratio approaches its historical mean
COST_BPS = 10         # Cost per unit of notional traded, per side

# 0 assumes month-t prices are available at month-end t.
# Use an appropriate lag if the data are published later.
PUBLICATION_LAG = 0

EU = "Natural gas, Europe"
US = "Natural gas, US"

# ------------------------------------------------------------
# 2. Prepare monthly data
# ------------------------------------------------------------
# Work with a subset of the data
df = data[["Year", "Month", EU, US]].copy()

df["Date"] = pd.to_datetime({
    "year": df["Year"],
    "month": df["Month"],
    "day": 1
})

df = df.set_index("Date")[[EU, US]].sort_index()
df.index = df.index.to_period("M")

# Prices should use comparable units and a common currency.
df["ratio"] = df[EU] / df[US]
df["log_ratio"] = np.log(df["ratio"])

# ------------------------------------------------------------
# 3. Historical normalisation: no future information
# ------------------------------------------------------------
# At t, the reference window ends at t-1.
past_ratio = df["log_ratio"].shift(1)

df["historical_mean"] = past_ratio.rolling(
    LOOKBACK, min_periods=LOOKBACK
).mean()

df["historical_std"] = past_ratio.rolling(
    LOOKBACK, min_periods=LOOKBACK
).std()

df["z_raw"] = (
    (df["log_ratio"] - df["historical_mean"])
    / df["historical_std"].replace(0, np.nan)
)

df["z_available"] = df["z_raw"].shift(PUBLICATION_LAG)

# ------------------------------------------------------------
# 4. Generate target positions at each month-end
# ------------------------------------------------------------
# +1 = buy Europe / sell US
# -1 = sell Europe / buy US
#  0 = flat
#
# Maintain the directional signal until convergence.
positions = []
position = 0

for z in df["z_available"]:
    if pd.isna(z):
        position = 0

    elif z >= ENTRY_Z:
        position = -1

    elif z <= -ENTRY_Z:
        position = 1

    elif position == 1 and z >= -EXIT_Z:
        position = 0

    elif position == -1 and z <= EXIT_Z:
        position = 0

    positions.append(position)

df["signal"] = positions

df["target_for_next_month"] = df["signal"].map({
    1: "BUY Europe / SELL US",
    -1: "SELL Europe / BUY US",
    0: "FLAT"
})

# ------------------------------------------------------------
# 5. Backtest: previous signal earns this month's return
# ------------------------------------------------------------
df["return_europe"] = df[EU].pct_change(fill_method=None)
df["return_us"] = df[US].pct_change(fill_method=None)

# Critical anti-look-ahead step:
df["position_held"] = df["signal"].shift(1).fillna(0)

# Allocate 50% of capital to each leg:
# 100% gross exposure, initially zero net exposure.
df["gross_return"] = (
    0.5 * df["position_held"]
    * (df["return_europe"] - df["return_us"])
)

# Explicit execution convention:
# Open each month's pair, then close both legs at month-end.
# If the signal persists, open a fresh pair for the next month.
# This also accounts for the final month's closing costs.
#
# Entry turnover = 0.5 + 0.5 = 1.
# Exit turnover reflects each leg's ending market value.
active = df["position_held"].abs()

df["turnover"] = active * (
    1.0
    + 0.5 * (1.0 + df["return_europe"])
    + 0.5 * (1.0 + df["return_us"])
)

df["cost"] = df["turnover"] * COST_BPS / 10_000
df["net_return"] = df["gross_return"] - df["cost"]

# Evaluate only months with a valid prior-month signal input.
eligible = df["z_available"].shift(1).notna()
bt = df.loc[eligible].copy()

if bt.empty:
    raise ValueError("No evaluable months.")

if (bt["net_return"] <= -1).any():
    raise ValueError(
        "The strategy loses all capital in a month. "
        "Reduce exposure and model margin/liquidation explicitly."
    )

# ------------------------------------------------------------
# 6. Performance statistics
# ------------------------------------------------------------
bt["wealth"] = (1 + bt["net_return"]).cumprod()

# Include initial wealth of 1 in the running peak.
peak = bt["wealth"].cummax().clip(lower=1)
bt["drawdown"] = bt["wealth"] / peak - 1

n_months = len(bt)
total_return = bt["wealth"].iloc[-1] - 1
annual_return = bt["wealth"].iloc[-1] ** (12 / n_months) - 1
annual_vol = bt["net_return"].std(ddof=1) * np.sqrt(12)

# Assumes zero cash return / risk-free rate.
monthly_vol = bt["net_return"].std(ddof=1)
sharpe = (
    np.sqrt(12) * bt["net_return"].mean() / monthly_vol
    if monthly_vol > 0 else np.nan
)

active_months = bt["position_held"].ne(0)
win_rate = (
    (bt.loc[active_months, "net_return"] > 0).mean()
    if active_months.any() else np.nan
)

print(f"Evaluation: {bt.index.min()} to {bt.index.max()}")
print(f"Months evaluated:       {n_months}")
print(f"Active months:          {active_months.sum()}")
print(f"Total return:           {total_return:.2%}")
print(f"Annualised return:      {annual_return:.2%}")
print(f"Annualised volatility:  {annual_vol:.2%}")
print(f"Sharpe (zero cash rate): {sharpe:.2f}")
print(f"Maximum drawdown:       {bt['drawdown'].min():.2%}")
print(f"Win rate, active months:{win_rate: .2%}")

# Signals are dated when generated, not when returns are earned.
print("\nLatest signals:")
print(
    df[["ratio", "z_available", "target_for_next_month"]]
    .tail(12)
    .round(3)
)

print(
    f"\nSignal at end of {df.index[-1]}, "
    f"for {df.index[-1] + 1}: "
    f"{df['target_for_next_month'].iloc[-1]}"
)

# ------------------------------------------------------------
# 7. Plot
# ------------------------------------------------------------
dates = df.index.to_timestamp()
test_dates = bt.index.to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axes[0].plot(dates, df["ratio"], color="navy")
axes[0].set_ylabel("Europe / US")
axes[0].set_title("Natural gas relative-value strategy")

axes[1].plot(
    dates, df["z_available"], color="purple", label="Available z-score"
)
axes[1].axhline(ENTRY_Z, color="red", linestyle="--")
axes[1].axhline(-ENTRY_Z, color="green", linestyle="--")
axes[1].axhline(0, color="grey", linewidth=0.8)
axes[1].set_ylabel("Z-score")
axes[1].legend()

axes[2].plot(
    test_dates, bt["wealth"],
    label="Pair strategy after illustrative costs"
)
axes[2].axhline(1, color="grey", linestyle="--", label="Zero-return cash")
axes[2].set_ylabel("Wealth, initial = 1")
axes[2].set_xlabel("Year")
axes[2].legend()

plt.tight_layout()

#Save figure
plt.savefig("Figure3.png")
plt.show()