"""
Trader class
"""

import random

class Trader():
    def __init__(self, capital):
        self.capital = capital
        self.stock = 0
        
    def trades(self, stock):
        for idx in range(len(stock.price)):
            if random.random() <= 0.5:
                if self.capital >= stock.price[idx]:
                    self.capital -= stock.price[idx]
                    self.stock += 1
            else:
                if self.stock >= 1:
                    self.capital += stock.price[idx]
                    self.stock -= 1
                    

                