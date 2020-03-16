from typing import TypeVar, Tuple, Mapping, Sequence
from mdp import MDP
from policy import Policy
from policy_evaluation import policyEvaluation
import numpy as np

S = TypeVar('S')
A = TypeVar('A')

def policyImprovementHelper(mdp : MDP,
                      state : S, 
                      value_for_policy : Mapping[S, float]) -> Sequence[float]:
    action_value_vector = np.zeros(len(mdp.actions))
    for action in mdp.actions:
        for next_state in mdp.states:
            if next_state in mdp.transition_data[state][action].keys():
                action_value_vector[action - 1] += mdp.transition_data[state][action][next_state] * \
                                (mdp.reward_data[state][action] + mdp.discount_factor * value_for_policy[next_state])
    return action_value_vector

def policyIteration(mdp : MDP, 
                    policy: Policy,
                    number_of_iterations : float) -> Tuple[Mapping[S, A], Mapping[S, float]]:
    count = 0
    optimal_policy = dict.fromkeys(mdp.non_terminal_states)
    while count < number_of_iterations:
        # evaluate current policy
        value_for_policy = policyEvaluation(mdp, policy, number_of_iterations)
        for state in mdp.non_terminal_states:
            # current action
            highest_prob_action = mdp.getPolicyForState()[state]
            # action through greedy search
            best_action = np.argmax(policyImprovementHelper(mdp, state, value_for_policy)) + 1
            if highest_prob_action != best_action:
                # update/improve policy
                optimal_policy[state] = best_action
        count += 1
    return [optimal_policy, value_for_policy]        

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
    print("Policy iteration: ", policyIteration(mdp, Policy(policy), 100))