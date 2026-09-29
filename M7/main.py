"""
First stock market simulation

@author: GK
"""

import matplotlib.pyplot as plt

from stock import Stock
from trader import Trader

         
#Initialise stock            
stock = Stock("A", 10)

#Simulate 1000 prices
stock.simulate(1000)

#Plot time series
plt.plot(stock.price)
plt.show()

#Trader
trader = Trader(100)

#Trades
trader.trades(stock)

#Capital and stock
print(f"The trader has {round(trader.capital,2)} and {trader.stock} shares.")