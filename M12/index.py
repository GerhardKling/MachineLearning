"""
Long-term stock market performance

"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


#Load data
path = r'C:\Users\Yunikarn LTD\Documents\Aberdeen\TEACHING\Machine Learning in Finance\Projects\L5 Pairs trading'
os.chdir(path)
data = pd.read_csv("stock_return.csv")

#Year and month
data["Year"] = np.floor(data["Date"]).astype(int)
data["Month"] = ((data["Date"]-data["Year"])*100).round().astype(int)

#Time variable for linear trend
data["Time"] = data["Year"] + (data["Month"] - 1) / 12

#Log values
data["Log_TRS"] = np.log(data["TRS"])

#Fit linear trend: log price = slope * time + intercept
slope, intercept = np.polyfit(data["Time"], data["Log_TRS"], 1)

#Visualisation: Cycles
plt.plot(data["Time"], data["Log_TRS"])
plt.plot(data["Time"], slope * data["Time"] + intercept, "r--", label="Linear trend")
plt.xlabel("Year")
plt.ylabel("Log real total returns")
plt.tight_layout()

#Save figure
plt.savefig("Figure1.png")
plt.show()

