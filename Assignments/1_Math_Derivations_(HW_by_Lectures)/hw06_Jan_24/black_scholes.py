import numpy as np
import math
from scipy.stats import norm

class black_scholes():
    def __init__(self, 
                 T : float, # final time
                 t : float, # current time
                 r : float, # risk free rate
                 S : float, # spot price
                 K : float, # strike price
                 sigma : float # volatility of returns
                 ):
        self.T = T
        self.t = t
        self.r = r
        self.tau = T - t
        self.S = S
        self.K = K
        self.sigma = sigma
        self.F = math.exp(r * self.tau) * S
        self.d_plus = 1 / (sigma * np.sqrt(self.tau)) * (np.log(self.F / K) + 1/2 * sigma ** 2 * self.tau)
        self.d_minus = self.d_plus - sigma * np.sqrt(self.tau)
        self.D = math.exp(-self.r * self.t)  # discount factor
        
    def callPrice(self):
        return self.D * (norm.cdf(self.d_plus, 0, 1) * self.F - norm.cdf(self.d_minus, 0, 1) * self.K)
    
    def putPrice(self):
        return self.D * (norm.cdf(-self.d_minus, 0, 1) * self.K - norm.cdf(-self.d_plus, 0, 1) * self.F)