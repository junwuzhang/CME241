from typing import TypeVar, Sequence, Mapping, Set, Tuple, Optional
import numpy as np
import random
from tabular_RL import tabularRL
from tabular_MC import tabularMC
from policy import Policy
from helper import getLambdaReturn, getSingleRV

S = TypeVar('S')
A = TypeVar('A')

class TDLambda():
    def __init__(
            self,
            tabular_RL : tabularRL,
            number_of_episodes : int,
            number_of_steps: int,
            learning_rate : float,
            learning_rate_decay : float,
            lambda_value : float):
        self.tabular_RL = tabular_RL
        self.number_of_episodes = number_of_episodes
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.lambda_value = lambda_value

    def forwardViewVFPrediction(self, policy : Policy):
        # offline
        value_function_prediction = {s: 0. for s in self.tabular_RL.state_action_relation.keys()}
        episodes = 0
        mc = tabularMC(self.tabular_RL, self.number_of_steps, self.number_of_episodes, True)
        
        while episodes < self.number_of_episodes:
            start_state = self.tabular_RL.generateState()
            mc_path = mc.getMCPath(policy, start_state)
            reward_vector = np.array([reward for _, _, reward, _ in mc_path[:-1]])
            state_vector = [state for state, _, _, _ in mc_path[:-1]]
            value_vector = np.array([value_function_prediction[s] for s in state_vector])
            if mc_path[-1][0] in self.tabular_RL.terminal_states:
                returns = getLambdaReturn(reward_vector, value_vector, \
                                        self.tabular_RL.gamma, self.lambda_value)
            else:
                raise RuntimeError('Has not terminated!')
            for t, _ in enumerate(returns):
                s, _, _, _ = mc_path[t]
                value_function_prediction[s] += self.learning_rate*(returns[t] - value_function_prediction[s])
            episodes += 1
            
        return value_function_prediction

    def backwardViewVFPrediction(self, policy : Policy):
        # online
        value_function_prediction = {s: 0. for s in self.tabular_RL.state_action_relation.keys()}
        act_gen_dict = {s: getSingleRV(policy.get_state_probabilities(s))
                        for s in self.tabular_RL.state_action_relation.keys()}
        episodes = 0
        updates = 0
        
        while episodes < self.number_of_episodes:
            eligibility_traces = {s: 0. for s in self.tabular_RL.state_action_relation.keys()}
            state = self.tabular_RL.generateState()
            steps = 0
            terminate = False
            
            while not terminate:
                action = act_gen_dict[state]()
                next_state, reward =\
                    self.tabular_RL.state_action_nextstate_reward_relation[state][action]()
                delta = reward + self.tabular_RL.gamma * value_function_prediction[next_state] -\
                    value_function_prediction[state]
                eligibility_traces[state] += 1
                alpha = self.learning_rate * (updates / self.learning_rate_decay
                                              + 1) ** -0.5
                for s in self.tabular_RL.state_action_relation.keys():
                    value_function_prediction[s] += alpha * delta * eligibility_traces[s]
                    eligibility_traces[s] *= (self.tabular_RL.gamma * self.lambda_value)
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
    lambda_value = 0.2

    tabular_RL = tabularRL(state_action_relation, state_action_nextstate_reward_relation, terminal_state, gamma)
    td = TDLambda(tabular_RL, number_of_episodes, number_of_steps, learning_rate, learning_rate_decay, lambda_value)
    print(td.forwardViewVFPrediction(Policy(policy)))
    print(td.backwardViewVFPrediction(Policy(policy)))


