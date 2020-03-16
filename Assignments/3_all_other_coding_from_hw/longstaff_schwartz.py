import math
import numpy as np 
from typing import TypeVar, Sequence, Mapping, Tuple
from numpy.linalg import inv

def longstaffSchwartz(SP : Sequence[Sequence],
                      payoff : Mapping[Tuple, float],
                      t : Sequence,
                      r : Sequence,
                      psi : Mapping[Tuple, float]):
    '''
    SP:     simulation paths: 2 dimensional array
    payoff: payoff array that can be used to search for option payoff at state s_ij
    t:      array of time stamps
    r:      array of infinitesimal risk-free rate at each time step 
    psi:    feature functions of state s_ij
    '''
    m = len(SP)
    n = len(SP[0]) - 1
    CF = [None] * m     # PV of current + future cashflow for path i
    for i in range(m):
        CF[i] = payoff[(t[n], SP[i, n + 1])]
    
    for j in range(n - 1, 1, -1):
        for i in range(m):
            CF[i] = CF[i] * math.exp(-r[t[j]] * (t[j + 1] - t[j]))
            s_ij = (t[j], SP[i, j + 1])
            if payoff[s_ij] > 0:
                X = psi[s_ij]
                Y = CF[i]
        w = inv(np.transpose(X).dot(X)).dot(np.transpose(X)).dot(Y)
        for i in range(m - 1):
            if payoff[s_ij] > np.transpose(w).dot(psi[s_ij]):
                CF[i] = payoff[s_ij]
    
    s_00 = (t[0], SP[0, 0 + 1])
    exercise_var = payoff[s_00]
    continue_var = math.exp(-r[0] * (t[1] - t[0])) * np.mean(CF)
    
    return (max(exercise_var, continue_var))
