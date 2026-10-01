"""
Yahoo Finance download

Installation on Anaconda >>> conda prompt
python -m pip install --upgrade yfinance
>>> Close your IDE and start again

"""

import yfinance as yf

#Download stock data
google = yf.download("GOOGL", 
                     start="2020-01-01", 
                     end="2026-10-01", 
                     auto_adjust=False, 
                     multi_level_index=False)

#Check data structure
print(google.head())

#Save to the current working directory
google.to_csv("Google_prices.csv")
