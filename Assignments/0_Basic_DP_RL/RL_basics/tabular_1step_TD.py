from typing import TypeVar, Sequence, Mapping, Set, Tuple, Optional
import numpy as np
import random
from tabular_RL import tabularRL
from policy import Policy
from helper import getSingleRV, getReturnsTerminating

class TD0():
    def __init__(self, 
                 tabular_RL : tabularRL, 
                 number_of_episodes : int, 
                 number_of_steps : int, 
                 learning_rate : float,
                 learning_rate_decay : float):
        self.tabular_RL = tabular_RL
        self.number_of_episodes = number_of_episodes
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

    def getValueFunctionPrediction(self, policy : Policy):
        value_function_prediction = {s: 0.0 for s in self.tabular_RL.state_action_relation.keys()}
        act_gen_dict = {s: getSingleRV(policy.get_state_probabilities(s))
                        for s in self.tabular_RL.state_action_relation.keys()}
        episodes = 0
        updates = 0

        while episodes < self.number_of_episodes:
            state = self.tabular_RL.generateState()
            steps = 0
            terminate = False
            while not terminate:
                action = act_gen_dict[state]()
                next_state, reward = self.tabular_RL.state_action_nextstate_reward_relation[state][action]()
                value_function_prediction[state] += self.learning_rate * \
                    (updates / self.learning_rate_decay + 1) ** (-0.5) * \
                    (reward + self.tabular_RL.gamma * value_function_prediction[next_state] - value_function_prediction[state])
                updates += 1
                steps += 1
                terminate = steps >= self.number_of_steps or state in self.tabular_RL.terminal_states
                state = next_state
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
    learning_rate = 0.1
    learning_rate_decay = 1e5
    
    tabular_RL = tabularRL(state_action_relation, state_action_nextstate_reward_relation, terminal_state, gamma)
    td = TD0(tabular_RL, number_of_episodes, number_of_steps, learning_rate, learning_rate_decay)
    val_func = td.getValueFunctionPrediction(Policy(policy))
