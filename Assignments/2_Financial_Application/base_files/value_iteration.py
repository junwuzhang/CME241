from typing import TypeVar, Tuple, Mapping, Sequence
from base_files.mdp import MDP
from base_files.policy import Policy
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

def valueIteration(mdp: MDP,
                   number_of_iterations : float) -> Tuple[Mapping[S, A], Mapping[S, float]]:
    count = 0
    optimal_policy = dict.fromkeys(mdp.non_terminal_states)
    final_value_function = mdp.getStateValueFunctions()
    current_action_value = 0
    while count <= number_of_iterations:
        for state in mdp.non_terminal_states:
            value = 0
            for action in mdp.actions:
                # action value function for action A
                for next_state in mdp.states:
                    if action not in mdp.transition_data[state].keys() or mdp.reward_data[state].keys():
                        continue
                    if next_state in mdp.transition_data[state][action].keys():
                        value += mdp.transition_data[state][action][next_state] * mdp.getStateValueFunctions()[next_state]
                        current_action_value = mdp.reward_data[state][action] + mdp.discount_factor * value
                    if current_action_value > final_value_function[state]:
                        final_value_function[state] = current_action_value
                        optimal_policy[state] = action
            count += 1 
    return [optimal_policy, final_value_function]

if __name__ == '__main__':
    transition = {
        1: {
            1: {1: 0.3, 2: 0.3, 3: 0.4},
            2: {1: 0.5, 2: 0.1, 3: 0.4},
        },
        2: {
            1: {1: 0.6, 2: 0.1, 3: 0.3},
            2: {1: 0.2, 2: 0.3, 3: 0.5},
        },
        3: {
            1: {3: 1.0},
            2: {3: 1.0},
        }
    }
    reward = {
        1: {1: 8.0, 2: 10.0},
        2: {1: 1.0, 2: -1.0},
        3: {1: 0.0, 2: 0.0}
    }
    policy = {
        1: {1: 0.5, 2: 0.5},
        2: {1: 0.2, 2: 0.8},
        3: {1: 1.0, 2: 0.0}
    }
    gamma = 0.8
    mdp = MDP(transition, reward, policy, gamma)
    print("Value iteration: ", valueIteration(mdp, 100))