"""
Momentum trading in commodities
pip install xgboost
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

path = r"C:\Users\Yunikarn LTD\Documents\Aberdeen\TEACHING\Machine Learning in Finance\Projects\L9 Trading stategies"
os.chdir(path)

from data import download
from trend import sma, ema, roc, rsi

#=========================================================================
#1. SAMPLE
#=========================================================================
#Commodities
tickers = [
    "GC=F", #Gold
    "SI=F", #Silver
    "HG=F", #Copper
    "PL=F", #Platinum
    "PA=F", #Palladium
    "ZC=F", #Corn
    "ZW=F", #Wheat
    "ZO=F", #Oats
    "ZR=F", #Rice
    "ZS=F"  #Soybeans
    ]

#=========================================================================
#2. DATA DOWNLOAD
#=========================================================================
try:
    df = pd.read_csv("Data.csv")
except:
    print("Data downloading...")
    download(tickers)
    df = pd.read_csv("Data.csv")
    print("Download completed.")
    
#=========================================================================
#3. DATA VISUALISATION
#=========================================================================
# Adjust date format
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.set_index("Date").sort_index()

# Keep dates with prices available for all stocks
df = df.dropna()

# Rebase each series to 100
indexed_prices = df.div(df.iloc[0]).mul(100)

# Plot using the Date index
indexed_prices.plot(figsize=(12, 6), linewidth=1.5)

plt.axhline(100, color="grey", linestyle="--", linewidth=0.8)
plt.title("Commodities: Indexed Daily Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price index (starting value = 100)")
plt.legend(title="Stock")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Line_chart.png")
plt.show()

#=========================================================================
#4. BUY & HOLD
#=========================================================================
# Annualised returns over five years (%)
annualised_returns = ((df.iloc[-1] / df.iloc[0]) ** (1 / 5) - 1) * 100

# Sort from best to worst performer
annualised_returns = annualised_returns.sort_values(ascending=False)

# Bar chart: column names become asset labels
annualised_returns.plot.bar(figsize=(12, 6), color="steelblue")

plt.axhline(0, color="grey", linewidth=0.8)
plt.title("Annualised Returns over Five Years")
plt.xlabel("Asset")
plt.ylabel("Annualised return (%)")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("Annualised_returns.png")
plt.show()

#=========================================================================
#5. TREND-FOLLOWING & MOMENTUM INDICATORS
#=========================================================================
# Target variable y and data matrix X
# Start with previous returns (lagged)
data = df.pct_change().shift(1)

# Caculate indicators for each asset and store in X
for asset in tickers:
    data[asset + "_sma"] = sma(df[asset])
    data[asset + "_ema"] = ema(df[asset])
    data[asset + "_roc"] = roc(df[asset])
    data[asset + "_rsi"] = rsi(df[asset])

# We try to predict gold price changes
data["y"] = df[tickers[0]].pct_change()

# Drop missing values and split into y and X
data = data.dropna()

y = data["y"]
X = data.drop("y", axis=1)

#=========================================================================
#6. FORECASTING MODELS
#=========================================================================
#PARAMETERS
k = 10 #Out-of-sample period in days


# X and y must already be aligned, ordered by date, and free of missing values
if len(X) != len(y):
    raise ValueError("X and y must have the same number of observations.")

if len(X) <= 10:
    raise ValueError("More than 10 observations are required.")

# Preserve the index for plotting
dates = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))

X_values = np.asarray(X, dtype=float)
y_values = np.asarray(y, dtype=float).reshape(-1)

if not np.isfinite(X_values).all() or not np.isfinite(y_values).all():
    raise ValueError("Remove missing or infinite values from X and y jointly.")

# Chronological split: final k observations are the test period
X_train, X_test = X_values[:-k], X_values[-k:]
y_train, y_test = y_values[:-k], y_values[-k:]

# Conservative starting settings; not tuned using the test period
models = {
    "OLS": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5,
        max_features=1.0,
        random_state=42,
        n_jobs=-1
        ),

    "XGBoost": XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        random_state=42,
        n_jobs=-1
        )
}

# Store actual values and forecasts
forecasts = pd.DataFrame(
    {"Actual": y_test},
    index=dates[-10:]
)

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    forecasts[name] = prediction

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, prediction),
        "RMSE": np.sqrt(mean_squared_error(y_test, prediction)),
        "Mean error": np.mean(prediction - y_test)
    })

# Rank models by RMSE: smaller is better
performance = (
    pd.DataFrame(results)
    .set_index("Model")
    .sort_values("RMSE")
)

print("\nOUT-OF-SAMPLE PERFORMANCE: FINAL 10 DAYS")
print(performance.round(6))

print("\nACTUAL VALUES AND FORECASTS")
print(forecasts.round(6))

# Plot actual values and forecasts
ax = forecasts.plot(figsize=(12, 6), marker="o", linewidth=1.5)
ax.lines[0].set_color("black")
ax.lines[0].set_linewidth(2.5)

plt.title("Out-of-Sample Forecasts: Final 10 Days")
plt.xlabel("Date")
plt.ylabel("Target variable")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Forecast_comparison.png")
plt.show()

#=========================================================================
#7. TRADING STRATEGY
#=========================================================================
# Strategy parameters
signal_threshold = 0.002    # Forecast must exceed +/-0.2%
transaction_cost = 0.0005  # 0.05% per unit of exposure traded, each way
initial_capital = 10000
allow_short = True
window = 300

# X and y must already be aligned and ordered chronologically
if isinstance(X, pd.DataFrame) and isinstance(y, (pd.Series, pd.DataFrame)):
    if not X.index.equals(y.index):
        raise ValueError("X and y must have matching indexes.")

dates = X.index if isinstance(X, pd.DataFrame) else pd.RangeIndex(len(X))

X_values = np.asarray(X, dtype=float)
y_values = np.asarray(y, dtype=float).reshape(-1)

if len(X_values) != len(y_values):
    raise ValueError("X and y must have the same number of observations.")

if len(y_values) <= window:
    raise ValueError("More than 100 observations are required.")

if not np.isfinite(X_values).all() or not np.isfinite(y_values).all():
    raise ValueError("Remove missing or infinite values from X and y jointly.")

predictions = []

for t in range(window, len(y_values)):

    # Most recent 100 observations, excluding the forecast target
    X_train = X_values[t - window:t]
    y_train = y_values[t - window:t]

    # One observation to forecast
    X_next = X_values[t:t + 1]

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions.append(model.predict(X_next)[0])

# All rolling out-of-sample forecasts
forecasts = pd.DataFrame({
    "Actual": y_values[window:],
    "XGBoost": predictions
}, index=dates[window:])

# Forecast accuracy over the entire evaluation period
errors = forecasts["XGBoost"] - forecasts["Actual"]

print(f"Out-of-sample observations: {len(forecasts)}")
print(f"MAE:  {errors.abs().mean():.6f}")
print(f"RMSE: {np.sqrt((errors ** 2).mean()):.6f}")
print(forecasts.head())


bt = forecasts[["Actual", "XGBoost"]].copy()
bt.columns = ["Actual_return", "Forecast_return"]

# Desired position for each forecast's holding period
bt["Position"] = 0
bt.loc[bt["Forecast_return"] > signal_threshold, "Position"] = 1

if allow_short:
    bt.loc[bt["Forecast_return"] < -signal_threshold, "Position"] = -1

bt["Signal"] = bt["Position"].map({
    1: "Long",
    0: "Cash",
    -1: "Short"
})

# Exposure traded: entry/exit = 1; long-to-short reversal = 2
previous_position = bt["Position"].shift(1, fill_value=0)
bt["Turnover"] = (bt["Position"] - previous_position).abs()

# Close any remaining position at the end of the backtest
bt.iloc[-1, bt.columns.get_loc("Turnover")] += abs(
    bt["Position"].iloc[-1]
)

bt["Cost"] = bt["Turnover"] * transaction_cost

# Actual_return is already the future return matched to each forecast:
# do NOT shift Position again.
bt["Gross_return"] = bt["Position"] * bt["Actual_return"]
bt["Net_return"] = bt["Gross_return"] - bt["Cost"]

# Buy-and-hold benchmark with entry and exit costs
bt["Buy_hold_return"] = bt["Actual_return"].copy()
bt.iloc[0, bt.columns.get_loc("Buy_hold_return")] -= transaction_cost
bt.iloc[-1, bt.columns.get_loc("Buy_hold_return")] -= transaction_cost

# Wealth paths
bt["Strategy_equity"] = initial_capital * (1 + bt["Net_return"]).cumprod()
bt["Buy_hold_equity"] = (
    initial_capital * (1 + bt["Buy_hold_return"]).cumprod()
)

# Compare performance over the test period
def performance_summary(returns):
    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax().clip(lower=1)
    drawdown = wealth / running_peak - 1

    return {
        "Total return (%)": 100 * (wealth.iloc[-1] - 1),
        "Maximum drawdown (%)": 100 * drawdown.min(),
        "Final capital": initial_capital * wealth.iloc[-1]
    }

summary = pd.DataFrame({
    "XGBoost strategy": performance_summary(bt["Net_return"]),
    "Buy and hold": performance_summary(bt["Buy_hold_return"])
}).T

print(summary.round(2))

print("\nSIGNALS AND RETURNS")
print(bt[[
    "Forecast_return", "Actual_return", "Signal",
    "Turnover", "Cost", "Net_return"
]].round(5))

# Plot
bt[["Strategy_equity", "Buy_hold_equity"]].plot(
    figsize=(12, 6), marker="o"
)
plt.axhline(initial_capital, color="grey", linestyle="--", linewidth=0.8)
plt.title("XGBoost Trading Strategy: Out-of-Sample Backtest")
plt.xlabel("Date")
plt.ylabel("Portfolio value")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("XGBoost_backtest.png")
plt.show()


#=========================================================================
#8. PERFORMANCE MEASURES
#=========================================================================
# Performance parameters
risk_free_rate = 0.05      # Annual effective risk-free rate
periods_per_year = 252     # Use 365 for daily crypto data

def performance_metrics(returns, risk_free_rate=0.05,
                        periods_per_year=252):

    returns = returns.dropna()
    n = len(returns)

    # Convert annual risk-free rate to daily rate
    rf_daily = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess_returns = returns - rf_daily

    # Cumulative wealth and compound annual growth rate
    wealth = (1 + returns).cumprod()
    total_return = wealth.iloc[-1] - 1
    cagr = wealth.iloc[-1] ** (periods_per_year / n) - 1

    # Annualised volatility and Sharpe ratio
    daily_volatility = returns.std(ddof=1)
    volatility = daily_volatility * np.sqrt(periods_per_year)

    sharpe = (
        excess_returns.mean() / daily_volatility
        * np.sqrt(periods_per_year)
        if daily_volatility > 0 else np.nan
    )

    # Sortino: downside deviations relative to the risk-free rate
    # Average across ALL days, including zero deviations on good days
    downside_deviation = np.sqrt(
        np.mean(np.minimum(excess_returns, 0) ** 2)
    )

    sortino = (
        excess_returns.mean() / downside_deviation
        * np.sqrt(periods_per_year)
        if downside_deviation > 0 else np.nan
    )

    # Drawdowns include starting wealth of 1
    running_peak = wealth.cummax().clip(lower=1)
    drawdown = wealth / running_peak - 1
    max_drawdown = -drawdown.min()  # Positive loss magnitude

    calmar = cagr / max_drawdown if max_drawdown > 0 else np.nan

    return pd.Series({
        "Total return (%)": total_return * 100,
        "CAGR (%)": cagr * 100,
        "Annualised volatility (%)": volatility * 100,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Maximum drawdown (%)": max_drawdown * 100,
        "Calmar ratio": calmar,
        "Positive days (%)": (returns > 0).mean() * 100,
        "Best day (%)": returns.max() * 100,
        "Worst day (%)": returns.min() * 100,
        "Observations": n
    })


performance = pd.DataFrame({
    "XGBoost strategy": performance_metrics(
        bt["Net_return"], risk_free_rate, periods_per_year
    ),
    "Buy and hold": performance_metrics(
        bt["Buy_hold_return"], risk_free_rate, periods_per_year
    )
})

print(performance.round(3))
