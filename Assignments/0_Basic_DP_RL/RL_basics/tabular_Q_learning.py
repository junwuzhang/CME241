from typing import TypeVar, Sequence, Mapping, Set, Tuple, Optional
import numpy as np
import random
from tabular_RL import tabularRL
from tabular_MC import tabularMC
from policy import Policy
from helper import getLambdaReturn, getSingleRV, getEpsilonGreedyAction, getExpectedActionValue

S = TypeVar('S')
A = TypeVar('A')

class tabularQLearning():
    def __init__(
        self,
        tabular_RL : tabularRL,
        number_of_episodes : int,
        number_of_steps: int,
        learning_rate : float,
        learning_rate_decay : float,
        epsilon : float) :
        self.tabular_RL = tabular_RL
        self.number_of_episodes = number_of_episodes
        self.number_of_steps = number_of_steps
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.epsilon = epsilon

    def getQValues(self):
        q_values = {state: {action: 0.0 for action in actions} \
                    for state, actions in self.tabular_RL.state_action_relation.items()}
        episodes = 0
        updates = 0
        
        while episodes < self.number_of_episodes:
            state = self.tabular_RL.generateState()
            action = getEpsilonGreedyAction(q_values[state], self.epsilon)
            steps = 0
            terminate = False
            
            while not terminate:
                next_state, reward = self.tabular_RL.state_action_nextstate_reward_relation[state][action]()
                next_action = getEpsilonGreedyAction(q_values[next_state], self.epsilon)
                # Q-Learning
                next_q_value = max(q_values[next_state][a] for a in q_values[next_state])
                q_values[state][action] += self.learning_rate *(updates / self.learning_rate_decay + 1) ** -0.5 *\
                    (reward + self.tabular_RL.gamma * next_q_value - q_values[state][action])
                updates += 1
                steps += 1
                terminate = steps >= self.tabular_RL.number_of_steps or state in self.tabular_RL.terminal_states
                state = next_state
                action = next_action
                
            episodes += 1
             
        return q_values

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
    epsilon = 0.2

    tabular_RL = tabularRL(state_action_relation, state_action_nextstate_reward_relation, terminal_state, gamma)
    sarsa = tabularQLearning(tabular_RL, number_of_episodes, number_of_steps, learning_rate, learning_rate_decay, epsilon)
    print(sarsa.getQValues())