"""
Classification with commodities

"""

import pandas as pd
import matplotlib.pyplot as plt


#Load data
data = pd.read_csv("Commodity.csv")

#Year and month
data["Year"] = data["Time"].str[:4].astype(int)
data["Month"] = data["Time"].str[-2:].astype(int)

#Visualisation: Cycles
plt.plot(data["Year"] + (data["Month"] - 1) / 12, data["Barley"])
plt.xlabel("Year")
plt.ylabel("Barley")
plt.tight_layout()

#Save figure
plt.savefig("Figure1.png")
plt.show()

#Ratio analysis
data["Co_Al"] =  data["Copper"]/data["Aluminum"]
plt.plot(data["Year"] + (data["Month"] - 1) / 12, data["Co_Al"])
plt.xlabel("Year")
plt.ylabel("Copper-Aluminium Ratio")
plt.tight_layout()

#Save figure
plt.savefig("Figure2.png")
plt.show()