from typing import TypeVar, Tuple, Mapping
from mdp import MDP
from policy import Policy
import numpy as np


S = TypeVar('S')
A = TypeVar('A')

def policyEvaluation(mdp: MDP, 
                     policy : Policy,
                     number_of_iterations : float) -> Mapping[S, float]:
    non_terminal_states = mdp.getNonterminalStates()
    policy_values = dict.fromkeys(non_terminal_states, 0)
    count = 0
    while count < number_of_iterations:
        for state in non_terminal_states:
            value = 0
            for action, probability in policy.pi[state].items():
                if action in mdp.transition_data[state].keys():
                    for next_state in mdp.states:
                        if next_state in mdp.transition_data[state][action].keys():
                            value += probability * mdp.transition_data[state][action][next_state] \
                                * (mdp.reward_data[state][action] + mdp.discount_factor * mdp.getStateValueFunctions()[next_state])
            policy_values[state] = value
        count += 1
    
    return policy_values

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
    print("Policy evaluation: ", policyEvaluation(mdp, Policy(policy), 100))
