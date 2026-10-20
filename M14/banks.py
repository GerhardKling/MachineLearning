"""
UK banks
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

#=========================================================================
#0. CHANGE DIRECTORY - OPTIONAL
#=========================================================================
path = r"C:\Users\Yunikarn LTD\Documents\Aberdeen\TEACHING\Machine Learning in Finance\Tutorials\Tutorial 3\Python"
os.chdir(path)

from data import download


#=========================================================================
#1. SAMPLE
#=========================================================================
#UK listed banks
tickers = [
    "BARC.L",   #Barclays
    "HSBA.L",   #HSBC Holdings
    "LLOY.L",   #Lloyds Banking Group
    "NWG.L",    #NatWest Group
    "STAN.L"    #Standard Chartered
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
#Keep dates with prices available for all five stocks
prices = df.dropna().drop("Date", axis=1).sort_index()

#Rebase each series to 100: divide by first price and multiply by 100
indexed_prices = prices.div(prices.iloc[0]).mul(100)

#Plot
indexed_prices.plot(figsize=(12, 6), linewidth=1.5)

plt.axhline(100, color="grey", linestyle="--", linewidth=0.8)
plt.title("UK banks: Indexed Daily Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price index (starting value = 100)")
plt.legend(title="Stock")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Line_chart.png")
plt.show()

#=========================================================================
#4. CORRELATION
#=========================================================================
#No obvious outlier
data = df.drop(["Date"], axis=1)
print(data.corr())

#=========================================================================
#5. LONG-RUN EQUILIBRIUM
#=========================================================================
#Focus on highly correlated assets STAN.L and HSBA.L
pair = df[["STAN.L", "HSBA.L"]].dropna().sort_index().copy()

#Estimate the long-run relationship using OLS
X = sm.add_constant(pair["STAN.L"])
model = sm.OLS(pair["HSBA.L"], X).fit()
print("\nLong-run Equilibrium")
print(model.summary())

#Store fitted equilibrium prices and residuals
pair["Equilibrium"] = model.fittedvalues
pair["ECT"] = model.resid

#Plot residuals over time
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(pair.index, pair["ECT"], linewidth=1, color="navy")
ax.axhline(0, color="black", linestyle="--", linewidth=0.8)

ax.set_title("Residuals from the Long-Run Regression")
ax.set_xlabel("Date")
ax.set_ylabel("Residual")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Residual.png")
plt.show()

#Residual-based Dickey–Fuller cointegration test
#trend="c": intercept in the long-run regression
#maxlag=0: no lagged differences in the residual DF regression
df_stat, p_value, critical_values = coint(
    pair["HSBA.L"],
    pair["STAN.L"],
    trend="c",
    maxlag=0,
    autolag=None
)

print("\nEngle–Granger residual-based DF test")
print("H0: No cointegration (residuals have a unit root)")
print(f"DF statistic: {df_stat:.4f}")
print(f"p-value:      {p_value:.4f}")

for level, value in zip(["1%", "5%", "10%"], critical_values):
    print(f"Critical value at {level}: {value:.4f}")

#=========================================================================
#6. VECM
#=========================================================================
#Create first differences
pair["D_HSBA.L"] = pair["HSBA.L"].diff()
pair["D_STAN.L"] = pair["STAN.L"].diff()

#Create lagged explanatory variables
pair["ECT_lag1"] = pair["ECT"].shift(1)
pair["D_HSBA.L_lag1"] = pair["D_HSBA.L"].shift(1)
pair["D_STAN.L_lag1"] = pair["D_STAN.L"].shift(1)

#Keep complete observations for both equations
regressors = ["ECT_lag1", "D_HSBA.L_lag1", "D_STAN.L_lag1"]

#New DataFrame for VECM
vecm_data = pair[["D_HSBA.L", "D_STAN.L"] + regressors].dropna().copy()

#Data matrix
X = sm.add_constant(vecm_data[regressors])

#Estimate the two equations using OLS
hsba_model = sm.OLS(vecm_data["D_HSBA.L"], X).fit()
stan_model = sm.OLS(vecm_data["D_STAN.L"], X).fit()

print("\nHSBA.L EQUATION")
print(hsba_model.summary())

print("\nSTAN.L EQUATION")
print(stan_model.summary())

#Compare coefficients across equations
coefficient_table = pd.DataFrame({
    "Delta HSBA.L": hsba_model.params,
    "Delta STAN.L": stan_model.params
    })

print("\nVECM COEFFICIENTS")
print(coefficient_table.round(3))

#Store fitted changes and equation residuals
vecm_data["D_HSBA.L_fitted"] = hsba_model.fittedvalues
vecm_data["D_STAN.L_fitted"] = stan_model.fittedvalues

vecm_data["HSBA.L_residual"] = hsba_model.resid
vecm_data["STAN.L_residual"] = stan_model.resid

#Report adjustment coefficients (speed of adjustment)
print("\nERROR CORRECTION COEFFICIENTS")
for name, result in [("HSBA.L", hsba_model), ("STAN.L", stan_model)]:
    print(
        f"{name}: alpha = {result.params['ECT_lag1']:.3f}, "
        f"p-value = {result.pvalues['ECT_lag1']:.3f}"
    )





