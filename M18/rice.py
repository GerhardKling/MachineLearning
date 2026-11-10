"""
Rice trading strategy
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#=========================================================================
#0. CHANGE DIRECTORY - OPTIONAL
#=========================================================================
path = r"C:\Users\Yunikarn LTD\Documents\Aberdeen\TEACHING\Machine Learning in Finance\Tutorials\Tutorial 4\Python"
os.chdir(path)

from trend import sma, ema, roc, rsi

#=========================================================================
#1. LOAD DATA
#=========================================================================
df = pd.read_csv("Data.csv")

    
#=========================================================================
#2. DATA VISUALISATION
#=========================================================================
# Adjust date format
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.set_index("Date").sort_index()

# Keep dates with prices available for all assets
df = df.dropna()

# Plot rice prices
df["ZR=F"].plot(figsize=(12, 6), linewidth=1.5)

plt.title("Rice prices")
plt.xlabel("Date")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Line_chart.png")
plt.show()


#=========================================================================
#3. JUMPS
#=========================================================================
returns = df.pct_change().dropna()

plt.boxplot(
    returns["ZR=F"],
    showfliers=True,
    whis=1.5,
    patch_artist=True,
    boxprops=dict(facecolor="lightblue"),
    medianprops=dict(color="black"),
    flierprops=dict(
        marker="o",
        markerfacecolor="red",
        markeredgecolor="red",
        markersize=4,
        alpha=0.6,
    ),
)

plt.xticks([1], ["Rice"])
plt.ylabel("Return")
plt.title("Rice returns and outliers")
plt.axhline(0, color="grey", linestyle="--", linewidth=0.8)
plt.tight_layout()
plt.savefig("Boxplot.png")
plt.show()

#Dummies for price jump due to rollover of contract
returns["Jump_up"] = (returns["ZR=F"] > 0.1).astype(int)
returns["Jump_down"] = (returns["ZR=F"] < -0.1).astype(int)

#Use dummies to identify jumps and adjust time series
data = returns[["ZR=F", "Jump_up", "Jump_down"]].dropna()

X = sm.add_constant(data[["Jump_up", "Jump_down"]])
model = sm.OLS(data["ZR=F"], X).fit()

returns["Rice_residual"] = model.resid

print(model.summary())


#=========================================================================
#4. INDICATORS
#=========================================================================
# Target variable y and data matrix X
# Caculate indicators for each asset and store in X
data = returns[["Rice_residual"]].rename(
    columns={"Rice_residual": "y"}
    ).copy()

# Calculate indicators using only y
data["SMA"] = sma(data["y"], 20)
data["EMA"] = ema(data["y"], 20)
data["ROC"] = roc(data["y"], 10)
data["RSI"] = rsi(data["y"], 14)

# Lag indicators to predict today's y using yesterday's information
indicator_cols = ["SMA", "EMA", "ROC", "RSI"]
data[indicator_cols] = data[indicator_cols].shift(1)

# Remove missing/infinite observations and align y and X
data = data.replace([float("inf"), -float("inf")], float("nan")).dropna()

y = data["y"]
X = data[indicator_cols]


#=========================================================================
#5. TRADING STRATEGY
#=========================================================================
# 1. Parameters
WINDOW = 200
SIGNAL_THRESHOLD = 0.001   # 0.1% predicted residual return
TRANSACTION_COST = 0.001   # 0.1% per unit of position change
RISK_FREE_RATE = 0.05
PERIODS_PER_YEAR = 252

# Assumes simple returns expressed as decimals.
# X already contains lagged indicators.
features = X.columns.tolist()

sample = pd.concat([
    y.rename("Target"),
    X,
    returns["ZR=F"].rename("Market_return"),
], axis=1)

sample = (
    sample.replace([np.inf, -np.inf], np.nan)
    .dropna()
    .sort_index()
)

if len(sample) <= WINDOW:
    raise ValueError("Need more than 200 complete observations.")


# 2. Rolling one-day-ahead forecasts
records = []

for t in range(WINDOW, len(sample)):

    train = sample.iloc[t - WINDOW:t]
    test = sample.iloc[[t]]

    X_train = train[features]
    y_train = train["Target"]
    X_test = test[features]

    # OLS
    ols = sm.OLS(
        y_train,
        sm.add_constant(X_train, has_constant="add"),
    ).fit()

    pred_ols = ols.predict(
        sm.add_constant(X_test, has_constant="add")
    ).iloc[0]

    # Random Forest
    forest = RandomForestRegressor(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=10,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )

    forest.fit(X_train, y_train)
    pred_rf = forest.predict(X_test)[0]

    records.append({
        "Date": sample.index[t],
        "Actual": test["Target"].iloc[0],
        "Market_return": test["Market_return"].iloc[0],
        "OLS_forecast": pred_ols,
        "RF_forecast": pred_rf,
    })

backtest = pd.DataFrame(records).set_index("Date")


# 3. Signals and net strategy returns
# +1 = long, 0 = cash, -1 = short
# Forecast for date t uses information available before t.
# Therefore, apply its signal directly to return t.
rf_daily = (1 + RISK_FREE_RATE)**(1 / PERIODS_PER_YEAR) - 1

for name in ["OLS", "RF"]:

    forecast = backtest[f"{name}_forecast"]

    signal = pd.Series(
        np.select(
            [
                forecast > SIGNAL_THRESHOLD,
                forecast < -SIGNAL_THRESHOLD,
            ],
            [1, -1],
            default=0,
        ),
        index=backtest.index,
    )

    # A long-to-short reversal has turnover of 2.
    turnover = signal.diff().abs()
    turnover.iloc[0] = abs(signal.iloc[0])

    backtest[f"{name}_signal"] = signal
    backtest[f"{name}_turnover"] = turnover

    # Simplified convention:
    # earn market return when long, its negative when short,
    # and the risk-free return when flat.
    backtest[f"{name}_return"] = (
        signal * backtest["Market_return"]
        + (signal == 0).astype(float) * rf_daily
        - TRANSACTION_COST * turnover
    )

# Buy and hold over exactly the same evaluation dates.
backtest["BH_return"] = backtest["Market_return"].copy()
backtest.iloc[
    0, backtest.columns.get_loc("BH_return")
] -= TRANSACTION_COST


# 4. Forecast accuracy
forecast_metrics = pd.DataFrame({
    name: {
        "RMSE": np.sqrt(mean_squared_error(
            backtest["Actual"], backtest[f"{name}_forecast"]
        )),
        "MAE": mean_absolute_error(
            backtest["Actual"], backtest[f"{name}_forecast"]
        ),
        "Direction accuracy": (
            np.sign(backtest["Actual"])
            == np.sign(backtest[f"{name}_forecast"])
        ).mean(),
    }
    for name in ["OLS", "RF"]
}).T

print("\nFORECAST PERFORMANCE")
print(forecast_metrics.round(4))


# 5. Trading performance

def performance_metrics(r):
    r = r.dropna()

    wealth = (1 + r).cumprod()

    # Include initial wealth of 1 when calculating drawdown.
    peak = wealth.cummax().clip(lower=1)
    drawdown = wealth / peak - 1

    annual_volatility = r.std(ddof=1) * np.sqrt(PERIODS_PER_YEAR)

    sharpe = (
        (r.mean() - rf_daily) / r.std(ddof=1)
        * np.sqrt(PERIODS_PER_YEAR)
        if r.std(ddof=1) > 0 else np.nan
    )

    cagr = (
        wealth.iloc[-1] ** (PERIODS_PER_YEAR / len(r)) - 1
        if (1 + r > 0).all() else np.nan
    )

    return {
        "Total return": wealth.iloc[-1] - 1,
        "CAGR": cagr,
        "Annual volatility": annual_volatility,
        "Sharpe ratio": sharpe,
        "Maximum drawdown": drawdown.min(),
        "Positive-day fraction": (r > 0).mean(),
    }


strategy_returns = backtest[
    ["OLS_return", "RF_return", "BH_return"]
].rename(columns={
    "OLS_return": "OLS",
    "RF_return": "Random Forest",
    "BH_return": "Buy and hold",
})

metrics = pd.DataFrame({
    name: performance_metrics(strategy_returns[name])
    for name in strategy_returns.columns
}).T

print(metrics.round(4))

print("\nTRADING PERFORMANCE — returns expressed as decimals")
print(metrics.round(4))


# 6. Cumulative performance
wealth = (1 + strategy_returns).cumprod()

wealth.plot(figsize=(12, 6))
plt.axhline(1, color="grey", linestyle="--", linewidth=0.8)
plt.title("Rolling 200-day forecasts: out-of-sample performance")
plt.ylabel("Growth of £1")
plt.xlabel("Date")
plt.tight_layout()
plt.savefig("Strategy.png")
plt.show()