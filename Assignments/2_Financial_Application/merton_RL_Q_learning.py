from typing import TypeVar, Sequence, Mapping, Set, Tuple
from scipy.linalg import eig
import numpy as np
import random
from base_files import tabular_RL
from base_files import tabular_Q_learning


S = Tuple[float, float]
A = Tuple[float, float]

state_action_relation_type = Mapping[S, Sequence[A]]
state_action_nextstate_reward_relation_type = Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]]
terminal_states_type = Sequence[S]

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
        return [state[0] + 1, next_wealth]

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
                         state: Tuple[float, float]) \
                             -> Tuple[state_action_relation_type, \
                                state_action_nextstate_reward_relation_type, terminal_states_type]:
        state_action_relation : state_action_relation_type = {}
        state_action_nextstate_reward_relation : state_action_nextstate_reward_relation_type = {}
        terminal_states : terminal_states_type = {}
        t = 0
        all_merton_states = []
        all_merton_states.append(state)
        while t < self.expiry:
            state = tuple(all_merton_states.pop(0))
            # assume for each state, there are two possible actions, each are generated randomly
            # this assumption is made so that we can simplify the MDP
            action_1 = tuple([random.uniform(0, 5), random.uniform(0, 5)])
            action_2 = tuple([random.uniform(0, 5), random.uniform(0, 5)])
            action_list = [action_1, action_2]
            state_action_relation[state] = action_list

            next_state_1 = tuple(self.getMertonTransition(state, action_1))
            next_state_2 = tuple(self.getMertonTransition(state, action_2))
            sub_dict_1 , sub_dict_2 = {}, {}
            # sub_dict : Mapping[S, Tuple[float, float]]
            sub_dict_1[next_state_1] = (0.5, self.getMertonReward(state))
            sub_dict_2[next_state_2] = (0.5, self.getMertonReward(state))
            
            state_action_nextstate_reward_relation[state] = {sub_dict_1, sub_dict_2}
            all_merton_states.append(next_state_1)
            all_merton_states.append(next_state_2)
            t += 0.1
        terminal_states = [state]
        return [state_action_relation, state_action_nextstate_reward_relation, terminal_states]

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
    state_action_relation, state_action_nextstate_reward_relation, terminal_states = mp.getMertonDataAll(initial_state)

    tabular_RL = tabular_RL.tabularRL(state_action_relation, state_action_nextstate_reward_relation, \
                                        terminal_states, discount_rate)
    

    