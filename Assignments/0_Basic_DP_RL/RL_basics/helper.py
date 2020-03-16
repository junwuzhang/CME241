from typing import TypeVar, Sequence, Mapping, Set, Tuple, \
                    Optional, Callable
from scipy.stats import rv_discrete
import numpy as np
from scipy.linalg import toeplitz
from operator import itemgetter

S = TypeVar('S')
A = TypeVar('A')

def getSingleRV(prob_dict: Mapping[S, float])\
        -> Callable[[], S]:
    outcomes, probabilities = zip(*prob_dict.items())
    rvd = rv_discrete(values=(range(len(outcomes)), probabilities))
    return lambda rvd=rvd, outcomes=outcomes: outcomes[rvd.rvs(size=1)[0]]

def getReturnsTerminating(
    rewards: Sequence[float],
    gamma: float) -> np.ndarray:
    sz = len(rewards)
    return toeplitz(
        np.insert(np.zeros(sz - 1), 0, 1.),
        np.power(gamma, np.arange(sz))
    ).dot(rewards)

def getLambdaReturn(reward_vector : Sequence[float], 
                       value_vector : Sequence[float],
                       gamma : float,
                       lambda_value : float):
    '''
    Based on lambda-return slide on David Silver's MC-TD slide
    '''
    T = len(reward_vector)
    discount_vector = np.power(gamma, np.arange(T + 1))
    returns = []
    for t in range(T):
        G_t = []
        for i in range(T - t):
            G_t.append(np.dot(reward_vector[t:t+i+1], discount_vector[:i+1]) \
                        + discount_vector[i+1] * value_vector[t+i]  )
        returns.append(sum([(1 - lambda_value) * (lambda_value ** n) * G for n , G in enumerate(G_t)]))
    return returns

def getEpsilonGreedyAction(action_value_dict: Mapping[A, float], 
                              epsilon: float):
    max_act = max(action_value_dict.items(), key=itemgetter(1))[0]
    if epsilon == 0:
        probability_dict = {max_act: 1.}
    else:
        probability_dict = {a: epsilon / len(action_value_dict) +
               (1. - epsilon if a == max_act else 0.)
               for a in action_value_dict.keys()}
    return getSingleRV(probability_dict)

def getEpsilonGreedyProbs(action_value_dict: Mapping[A, float], 
                             epsilon: float):
    max_act = max(action_value_dict.items(), key=itemgetter(1))[0]
    if epsilon == 0:
        probability_dict = {max_act: 1.}
    else:
        probability_dict = {a: epsilon / len(action_value_dict) +
               (1. - epsilon if a == max_act else 0.)
               for a in action_value_dict.keys()}
    return probability_dict

def getExpectedActionValue(
    action_value_dict: Mapping[A, float],
    epsilon: float) -> float:
    av = action_value_dict
    ap = getEpsilonGreedyProbs(av, epsilon)
    return sum(ap.get(a, 0.) * v for a, v in av.items())