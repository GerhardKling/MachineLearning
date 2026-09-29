"""
Stock class
"""

import random

class Stock():
    def __init__(self, name, P0):
        self.name = name
        self.price = [P0]
    
    def simulate(self, num):
        while num:      
            if random.random() <= 0.5:
                price = self.price[-1] - 0.1
            else:
                price = self.price[-1] + 0.1
            num -= 1
            self.price.append(round(price,2))