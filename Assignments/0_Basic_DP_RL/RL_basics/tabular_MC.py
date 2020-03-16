from typing import TypeVar, Sequence, Mapping, Set, Tuple, Optional
import numpy as np
import random
from tabular_RL import tabularRL
from policy import Policy
from helper import getSingleRV, getReturnsTerminating

S = TypeVar('S')
A = TypeVar('A')

class tabularMC():
    def __init__(self, 
                 tabular_RL: tabularRL, 
                 number_of_steps: int, 
                 number_of_episodes : int, 
                 first_visit : bool):
        self.tabular_RL = tabular_RL
        self.number_of_steps = number_of_steps
        self.number_of_episodes = number_of_episodes
        self.first_visit = first_visit

    def getMCPath(
        self,
        policy : Policy,
        start_state : S,
        start_action : Optional[A] = None) -> Sequence[Tuple[S, A, float, bool]]:

        res = []
        state = start_state
        steps = 0
        occ_states = set()
        act_gen_dict = {s: getSingleRV(policy.get_state_probabilities(s))
                        for s in self.tabular_RL.state_action_relation.keys()}

        while state not in self.tabular_RL.terminal_states\
            and steps <= self.number_of_steps:
            first_visit_state = state not in occ_states
            occ_states.add(state)
            action = act_gen_dict[state]()\
                if (steps > 0 or start_action is None) else start_action
            next_state, reward =\
                self.tabular_RL.state_action_nextstate_reward_relation[state][action]()
            res.append((state, action, reward, first_visit_state))
            steps += 1
            state = next_state
        return res

    def getValueFunctionPrediction(self, policy: Policy) -> Mapping[S, float]:
        counts_dict = {s: 0 for s in self.tabular_RL.state_action_relation.keys()}
        value_function_prediction = {s: 0.0 for s in self.tabular_RL.state_action_relation.keys()}
        episodes = 0
        
        while episodes < self.number_of_episodes:
            start_state = self.tabular_RL.generateState()
            mc_path = self.getMCPath(policy, start_state)
            reward_vector = np.array([reward for _, _, reward, _ in mc_path[:-1]])
            if mc_path[-1][0] in self.tabular_RL.terminal_states:
                returns = getReturnsTerminating(reward_vector, self.tabular_RL.gamma)
            else:
                raise RuntimeError('Has not terminated!')
            for t, r in enumerate(returns):
                s, _, _, f = mc_path[t]
                if not self.first_visit or f:
                    counts_dict[s] += 1
                    c = counts_dict[s]
                    value_function_prediction[s] = (value_function_prediction[s] * (c - 1) + r) / c
            episodes += 1
        return value_function_prediction

if __name__ == '__main__':
    # model parameters are same as those used in DP algorithms
    state_action_relation = {1: [1, 2], 2: [1, 2], 3: [1, 2]}
    terminal_state = [3]
    state_action_nextstate_reward_relation = {
        1: {
            1: {
                1: (0.3, 5.0), 2: (0.3, 5.0), 3: (0.4, 5.0)
            },
            2: {
                1: (0.5, 15.0), 2: (0.1, 15.0), 3: (0.4, 15.0)
            }
        },
        2: {
            1: {
                1: (0.6, 3.0), 2: (0.1, 3.0), 3: (0.3, 3.0)
            },
            2: {
                1: (0.2, -3.0), 2: (0.3, -3.0), 3: (0.5, -3.0)
            }
        },
        3: {
            1: {
                3: (1.0, 0.0)}, 2: {3: (1.0, 0.0)
            }
        }
    }
    policy = {
        1: {
            1: 0.5, 2: 0.5
        }, 
        2: {
            1: 0.2, 2: 0.8
        }, 
        3: {
            1: 1.0, 2: 0.0
        }
    }
    
    gamma = 0.8
    number_of_episodes = 100
    number_of_steps = 100
    
    tabularRL = tabularRL(state_action_relation, state_action_nextstate_reward_relation, terminal_state, gamma)
    mc = tabularMC(tabularRL, number_of_episodes, number_of_steps, True)
    val_func = mc.getValueFunctionPrediction(policy)

    