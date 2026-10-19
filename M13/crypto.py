"""
Momentum in crypto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data import download
from trend import sma, ema, roc, rsi, macd

#=========================================================================
#1. SAMPLE
#=========================================================================
#US listed steelmakers by market cap
tickers = [
    "BTC-USD",   # Bitcoin-USD
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
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values("Date")

plt.figure(figsize=(10, 5))
plt.plot(df["Date"], df["BTC-USD"])
plt.xlabel("Time")
plt.ylabel("Bitcoin price")
plt.title("Bitcoin price in USD")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Line_chart.png")
plt.show()

#=========================================================================
#4. TREND-FOLLOWING & MOMENTUM INDICATORS
#=========================================================================
# Handle missing prices before calculating the indicators.
data = df.dropna(subset=["BTC-USD"]).copy()

price = data["BTC-USD"]

data["SMA_20"] = sma(price, window=20)
data["EMA_20"] = ema(price, window=20)
data["ROC_12"] = roc(price, window=12)
data["RSI_14"] = rsi(price, window=14)

data[["MACD", "MACD_signal", "MACD_histogram"]] = macd(price)

#=========================================================================
#5. TRADING SIGNALS
#=========================================================================
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(12, 7),
    sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}
)

#Bitcoin price
ax1.plot(data["Date"], data["BTC-USD"], color="steelblue")
ax1.set_title("Bitcoin price and Relative Strength Index")
ax1.set_ylabel("BTC price (USD)")
ax1.grid(alpha=0.3)

#RSI
ax2.plot(data["Date"], data["RSI_14"], color="purple", label="RSI (14)")

ax2.axhline(70, color="red", linestyle="--", label="Upper threshold: 70")
ax2.axhline(30, color="green", linestyle="--", label="Lower threshold: 30")
ax2.axhline(50, color="grey", linestyle=":", alpha=0.5)

#Highlight threshold exceedances.
ax2.fill_between(
    data["Date"], 70, data["RSI_14"],
    where=data["RSI_14"] > 70,
    interpolate=True,
    color="red", alpha=0.35
)

ax2.fill_between(
    data["Date"], data["RSI_14"], 30,
    where=data["RSI_14"] < 30,
    interpolate=True,
    color="green", alpha=0.35
)

ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("RSI_chart.png")
plt.show()

#=========================================================================
#6. DIMENSIONALITY REDUCTION
#=========================================================================
#Select features
X = (
    data.iloc[:, 2:]
    .select_dtypes(include="number")
    .replace([np.inf, -np.inf], np.nan)
    .dropna()
    )

print("Features included:", X.columns.tolist())

#Standardise: each feature has mean 0 and variance 1.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Fit PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_scaled)

#Explained variance
explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
components = np.arange(1, len(explained) + 1)

plt.figure(figsize=(9, 5))

plt.bar(
    components, explained,
    alpha=0.5, align="center",
    label="Individual explained variance"
    )

plt.step(
    components, cumulative,
    where="mid",
    label="Cumulative explained variance"
    )

plt.xticks(components)
plt.ylim(0, 1.05)
plt.ylabel("Explained variance ratio")
plt.xlabel("Principal component")
plt.legend()
plt.tight_layout()
plt.savefig("PCA_chart.png")
plt.show()

#Two principal components are sufficient
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)










