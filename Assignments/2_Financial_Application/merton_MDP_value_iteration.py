from typing import TypeVar, Sequence, Mapping, Set, Tuple
from scipy.linalg import eig
from base_files import mdp
from base_files import value_iteration
import numpy as np
import random
import copy
import math

S = Tuple[float, float]
A = Tuple[float, float]

trans_matrix_type = Mapping[S, Mapping[A, Mapping[S, float]]]
reward_type = Mapping[S, Mapping[A, float]]
policy_type = Mapping[S, Mapping[A, float]]

class mertonPortofolio():
    def __init__(self, 
                 expiry : float, 
                 r : float, 
                 mu : np.ndarray,
                 cov : np.ndarray,
                 epsilon: float,
                 gamma: float):
        self.expiry = expiry        # = T
        self.r = r                  # = risk-free rate
        self.mu = mu                # = risky rate means (1-D array of length num risky assets)
        self.cov = cov              # = risky rate covariances (2-D square array of length num risky assets)
        self.epsilon = epsilon      # = bequest parameter
        self.gamma = gamma          # = CRRA parameter

    def getMertonTransition(self, 
                            state: Tuple[float, float], 
                            action: Tuple[float, float]):
        risky_return = np.random.normal(self.mu[0], self.cov[0])
        wealth = state[1]
        risky_allocation = action[0]
        wealth_consumption = action[1]
        next_wealth = (wealth - wealth_consumption) * \
                    ((1 - risky_allocation) * (1 + self.r) + risky_allocation * (1 + risky_return))
        return [state[0] + 1, next_wealth[0]]

    def getMertonReward(self, state: S) ->float :
        time = state[0]
        wealth = state[1]
        if time != self.expiry:
            if self.gamma == 0:
                return np.log(wealth)
            else:
                return wealth ** (1.0-self.gamma) / (1.0-self.gamma)
        else:
            return 0.0

    def getMertonDataAll(self,
                         state: Tuple[float, float]) -> Tuple[trans_matrix_type, reward_type, policy_type]:
        transition_matrix : trans_matrix_type = {}
        reward : reward_type = {}
        policy : policy_type = {}
        t = 0
        all_merton_states = []
        all_merton_states.append(state)
        while t < self.expiry:
            state = tuple(all_merton_states.pop(0))
            # assume for each state, there are two possible actions, each are generated randomly
            # this assumption is made so that we can simplify the MDP
            action_1 = tuple([random.uniform(0, 5), random.uniform(0, 5)])
            action_2 = tuple([random.uniform(0, 5), random.uniform(0, 5)])
            next_state_1 = tuple(self.getMertonTransition(state, action_1))
            next_state_2 = tuple(self.getMertonTransition(state, action_2))
            sub_dict_1 = {action_1: {next_state_1: 1.0}}
            sub_dict_2 = {action_2: {next_state_2: 1.0}} 
            if state in transition_matrix.keys():
                list1 = list(transition_matrix.items())
                list2 = list(sub_dict_1.items())
                list1[0][1][list2[0][0]] = list2[0][1]
                transition_matrix = dict(list1)
                list1 = list(transition_matrix.items())
                list2 = list(sub_dict_2.items())
                list1[0][1][list2[0][0]] = list2[0][1]
                transition_matrix = dict(list1)
            else:
                transition_matrix[state] = copy.deepcopy(sub_dict_1)
                list1 = list(transition_matrix.items())
                list2 = list(sub_dict_2.items())
                list1[0][1][list2[0][0]] = list2[0][1]
                transition_matrix = dict(list1)
            reward[state] = {action_1: self.getMertonReward(state), 
                             action_2: self.getMertonReward(state)}
            policy[state] = {action_1: 0.5, action_2: 0.5}
            all_merton_states.append(next_state_1)
            all_merton_states.append(next_state_2)
            t += 0.1
        return [transition_matrix, reward, policy]

if __name__ == '__main__':
    expiry = 0.4
    r = 0.04
    mu = np.array([0.08])
    cov = np.array([[0.0009]])
    epsilon = 1e-8
    gamma = 0.2
    discount_rate = 0.8

    mp = mertonPortofolio(expiry, r, mu, cov, epsilon, gamma)

    initial_wealth = 10
    initial_state = [0, initial_wealth]
    transition_matrix, reward, policy = mp.getMertonDataAll(initial_state)
    # print(policy)

    mdp = mdp.MDP(transition_matrix, reward, policy, discount_rate)
    vi_dict = value_iteration.valueIteration(mdp, 100)[0]
    new_dict = {}
    for key, val in vi_dict.items():
        if val != None and val != float('nan'):
            new_dict[key] = val
    print("Value iteration: ", new_dict)

