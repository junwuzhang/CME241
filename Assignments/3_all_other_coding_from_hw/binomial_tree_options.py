import numpy as np
import math

def americanOption(T, S, K, r, sigma, q, n, a):
    '''
    T... expiration time
    S... stock price
    K... strike price
    q... dividend yield
    n... height of the binomial tree
    a... action: put or call?
    '''

    # creation of binomial price tree
    deltaT = T / n
    up = math.exp(sigma * np.sqrt(deltaT))
    p0 = (up * math.exp(-q * deltaT) - math.exp(-r * deltaT)) / (up**2 - 1)
    p1 = math.exp(-r * deltaT) - p0
    p = []

    # calculation of option value at each final node
    for i in range(n):
        if a is 'put':
            option_value = K - S * up**(2*i - n)
        else:
            option_value = S * up**(2*i - n) - K
        p[i] = option_value
        if p[i] < 0:
            p[i] = 0

    # sequential calculation of option value at each preceding node
    for j in range(n-1):
        for i in range(j):
            p[i] = p0 * p[i+1] + p1 * p[i]     # binomial value
            if a is 'put':
                exercise = K - S * up**(2*i - j)    # exercise value
            else:
                exercise = S * up**(2*i - j) - K
            if p[i] < exercise:
                p[i] = exercise

    return p[0]
