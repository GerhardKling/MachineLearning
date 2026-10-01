"""
Data Analysis

"""
import os
import pandas as pd
import matplotlib.pyplot as plt

#Data import with change of directory
#Or import from current working directory
path = r"C:\Users\Yunikarn LTD\Documents\Aberdeen\TEACHING\Machine Learning in Finance\Projects\L3 Data_analysis"
os.chdir(path)

data = pd.read_csv("UK_gold.csv")

#Create year and quarter variable
#Strings such as "2024 Q1"
data["Year"] = data["Time"].str[:4].astype(int)
data["Quarter"] = data["Time"].str[-1].astype(int)

#Visualisation
plt.plot(data["Year"] + (data["Quarter"] - 1) / 4, data["Holding"])
plt.xlabel("Year")
plt.ylabel("Holding")
plt.tight_layout()

#Save figure
plt.savefig("Figure1.png")
plt.show()

#Gold prices
data_price = pd.read_csv("Gold_price.csv")

#Create year and month variable
#Strings such as "2024 Q1"
data_price["Year"] = data_price["Time"].str[:4].astype(int)
data_price["Month"] = data_price["Time"].str[-2:].astype(int)

#Create quarter: Jan–Mar = 1, Apr–Jun = 2, etc.
#Division // ignoring remainder
data_price["Quarter"] = (data_price["Month"] - 1) // 3 + 1

#Calculate the mean price for each year and quarter
quarterly_prices = (
    data_price.groupby(["Year", "Quarter"], as_index=False)["Gold"]
    .mean()
)

#Merge gold holdings and quarterly prices
merged_data = pd.merge(
    data,
    quarterly_prices,
    on=["Year", "Quarter"],
    how="inner",
    validate="one_to_one"
)

#Check merged data
print(merged_data.head())







