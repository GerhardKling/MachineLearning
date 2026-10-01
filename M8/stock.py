"""
Stock class
Log returns: P_t = P_{t-1}e^r thus \ln P_t = \ln P_{t-1} + r
"""

import random
import numpy as np

class Stock():
    def __init__(self, name, P0):
        self.name = name
        self.price = [P0]
    
    def simulate(self, num):
        while num:      
            if random.random() <= 0.5:
                price = np.log(self.price[-1]) - 0.01
            else:
                price = np.log(self.price[-1]) + 0.01
            num -= 1
            self.price.append(round(np.exp(price),2))
            
