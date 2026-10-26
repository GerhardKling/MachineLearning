"""
Volatility trading

For ARCH/GARCH models
Run once in Anaconda Prompt:
conda install -c conda-forge arch-py

Or in Spyder terminal
pip install arch
Then start new terminal or go to Consoles to restart kernel

For Excel files
pip install openpyxl

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.stats.diagnostic import het_arch
from arch import arch_model

from data import download
from tables import coefficient_table

#=========================================================================
#1. DATA
#=========================================================================
ticker = ['TSLA']

try:
    df = pd.read_csv('Data.csv')
except:
    download(ticker)
    df = pd.read_csv('Data.csv')
    
#=========================================================================
#2. VOLATILITY CLUSTERS
#=========================================================================
#Stock return
df['Return'] = df['TSLA'].pct_change()

#Drop missing value
df = df.dropna()

#Visualisation of volatility clusters
df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
df = df.sort_values("Date")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df["Date"], df["Return"], linewidth=1)

#Automatically choose readable date intervals and labels
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_xlabel("Date")
ax.set_ylabel("Return")
ax.set_title("Returns over time")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Clusters.png")
plt.show()

#=========================================================================
#3. ENGLE'S ARCH LM TEST
#=========================================================================
#Residuals from a constant mean return model (demean returns)
returns = df["Return"]
residuals = returns - returns.mean()

#Test for ARCH effects for different lags
print("ARCH LM TEST")
print("\nReject H0: Evidence of ARCH effects (volatility clustering).")
    
for lag in [5, 10, 20]:
    lm_stat, p_value, _, _ = het_arch(residuals, nlags=lag)
    print(f"Lags: {lag:2d} | LM: {lm_stat:.3f} | p-value: {p_value:.4f}")
    
#=========================================================================
#4. ARCH/GARCH
#=========================================================================
#Convert decimal returns to percentage returns for numerical stability
returns = df["Return"].dropna() * 100

#ARCH(10): ten lagged squared shocks
arch_model_10 = arch_model(
    returns,
    mean="Constant",
    vol="ARCH",
    p=10,
    dist="normal"
)
arch_result = arch_model_10.fit(disp="off")
print("\nARCH(10)")
print(arch_result.summary())

# GARCH(1,1): one lagged squared shock and one lagged variance
garch_model_11 = arch_model(
    returns,
    mean="Constant",
    vol="GARCH",
    p=1,
    q=1,
    dist="t"       # Student's t innovations allow heavier tails
)
garch_result = garch_model_11.fit(disp="off")

print("\nGARCH(1,1)")
print(garch_result.summary())

#=========================================================================
#5. Excel tables
#=========================================================================
# Model comparison: lower AIC/BIC indicates a better fit
# after penalising model complexity
comparison = pd.DataFrame({
    "ARCH(5)": {
        "Observations": arch_result.nobs,
        "Log likelihood": arch_result.loglikelihood,
        "AIC": arch_result.aic,
        "BIC": arch_result.bic
    },
    "GARCH(1,1)": {
        "Observations": garch_result.nobs,
        "Log likelihood": garch_result.loglikelihood,
        "AIC": garch_result.aic,
        "BIC": garch_result.bic
    }
}).rename_axis("Statistic")


# Save to the current working directory
with pd.ExcelWriter("ARCH_GARCH_results.xlsx", engine="openpyxl") as writer:
    coefficient_table(arch_result).to_excel(writer, sheet_name="ARCH")
    coefficient_table(garch_result).to_excel(writer, sheet_name="GARCH")
    comparison.to_excel(writer, sheet_name="Model comparison")

    # Adjust column widths and display precision
    for sheet in writer.book.worksheets:
        sheet.freeze_panes = "B2"
        sheet.column_dimensions["A"].width = 22

        for column in sheet.iter_cols(min_col=2):
            sheet.column_dimensions[column[0].column_letter].width = 20
            for cell in column[1:]:
                cell.number_format = "0.0000"

print("Saved: ARCH_GARCH_results.xlsx")


#=========================================================================
#6. Forecasting
#=========================================================================
#One-day-ahead forecast
forecast = garch_result.forecast(horizon=1, reindex=False)

next_variance = forecast.residual_variance.iloc[-1, 0]
next_volatility = np.sqrt(next_variance)

print(f"Next-day conditional volatility: {next_volatility:.3f}%")

# 2. Match dates to the observations used in estimation
dates = pd.to_datetime(
    df.loc[df["Return"].notna(), "Date"]
    )

volatility = pd.Series(
    garch_result.conditional_volatility.to_numpy(),
    index=pd.DatetimeIndex(dates),
    name="Conditional volatility"
    )

#For stocks: next weekday (does not account for exchange holidays)
next_date = volatility.index[-1] + pd.offsets.BDay(1)

#For crypto, use the next calendar day instead:
#next_date = volatility.index[-1] + pd.Timedelta(days=1)

#Plot recent fitted volatility and the forecast
recent = volatility.tail(100)

fig, ax = plt.subplots(figsize=(11, 5))

ax.plot(
    recent.index, recent,
    color="steelblue",
    label="Fitted conditional volatility"
    )

ax.plot(
    [recent.index[-1], next_date],
    [recent.iloc[-1], next_volatility],
    color="crimson", linestyle="--"
    )

ax.scatter(
    next_date, next_volatility,
    color="crimson", s=65, zorder=5,
    label=f"Next-day forecast: {next_volatility:.2f}%"
    )

locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.set_title("GARCH(1,1): one-day-ahead volatility forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Daily conditional volatility (%)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("Forecast.png")
plt.show()