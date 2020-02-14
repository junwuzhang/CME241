from typing import TypeVar, Sequence, Mapping, Set, Tuple
import numpy as np
from scipy.linalg import eig
from mrp import MRP
from policy import Policy

S = TypeVar('S')
A = TypeVar('A')

"""
Markov Decision Process is a Markov Reward Process with decisions
MDP is a tuple <S, A, P, R, gamma> where:
S is a finite set of states
A is a finite set of actions
P is a state transition probability matrix
R is a reward function: Rs = E[R_{t+1} | S_t = S]
gamma is a discount factor
"""

class MDP(MRP):
    def __init__(self, 
                transition_data: Mapping[S, Mapping[A, Mapping[S, float]]],
                reward_data: Mapping[S, Mapping[A, float]], 
                policy_data: Mapping[S, Mapping[A, float]],
                gamma: float):
        # transition data: in the form of: Mapping[S, Mapping[A, Mapping[next_S, prob]]]
        # this is also a transtion_matrix
        self.transition_data : Mapping[S, Mapping[A, Tuple[Mapping[S, float], float]]] = transition_data
        # reward data: in the form of: Mapping[S, Mapping[A, reward]]
        self.reward_data : Mapping[S, Mapping[A, float]] = reward_data
        # policy data: in the form of: Mapping[S, Mapping[A, probability]]
        self.policy_data : Mapping[S, Mapping[A, float]] = policy_data
        self.states : Sequence[S] = self.transition_data.keys()
        self.actions : Sequence[A] = self.getActions()
        self.state_actions_relations : Mapping[S, Sequence[A]] = self.getActionsForState()
        self.state_value_functions : Mapping[S, float] = self.getInitialStateValueFunctions()
        self.action_value_functions : Mapping[S, Mapping[A, float]] = self.getInitialActionValueFunctions()
        self.discount_factor : float = gamma
        # self.terminal_states = self.getTerminalStates()
        # self.non_terminal_states = self.getNonterminalStates()
        # self.transition_matrix : np.ndarray = self.getTransitionMatrix()   # specific to MRP
        # self.gamma : float = gamma
        # self.reward_vector = self.getRewardVector()

    def getActions(self) -> Sequence[A]:
        actions = []
        for data in self.transition_data.items():
            for a in data.keys():
                actions.append(a)
        return [set(actions)]

    def getActionsForState(self) -> Mapping[S, Sequence[A]]:
        actions_for_state = {}
        possible_actions = set()
        for state, data in self.transition_data.items():
            possible_actions.add(data.keys())
            actions_for_state[state] = [possible_actions]
        return actions_for_state

    def getTransitionMatrix(self) -> Mapping[S, Mapping[A, Mapping[S, float]]]:
        return self.transition_data     

    def getTerminalStates(self) -> Sequence[S]:
        terminal_states = []
        for s, trans_data in self.transition_data.items():
            if len(trans_data) == 1 and s in trans_data.keys():
                terminal_states.append(s)
        return terminal_states

    def getNonterminalStates(self) -> Sequence[S]:
        return list(set(self.states) - set(self.terminal_states))

    # TODO: fix everything below to be working code once I have time
    def getPolicyForState(self) -> Mapping[S, A]:
        policy_for_state = {}
        for s, action_with_prob in self.policy_data:
            highest_prob_action = np.random.choice(self.actions, p=[action_with_prob.items()])
            policy_for_state[s] = highest_prob_action
        return policy_for_state

    def toMRP(self) -> MRP:
        # transiton_data_mrp : Mapping[S, [next_S, probability]]
        transition_data_mrp = {}
        # rewards_data_mrp : Mapping[S, reward]
        rewards_data_mrp: {}
        # rewards = {1: 5.0, 2: 2.0, 3: 1.0}
        for s, data in self.transition_data:
            for state, action in self.getPolicyForState():
                transition_data_mrp[s] = self.transition_data[s][action]
        for s, data in self.reward_data:
            for state, action in self.getPolicyForState():
                rewards_data_mrp[s] = self.reward_data[s][action]
        mrp = MRP(transition_data_mrp, self.discount_factor)
        mrp.setRewardFunctionDeterministic(rewards_data_mrp)
        return mrp        

    def getInitialStateValueFunctions(self) -> Mapping[S, float]:
        return dict.fromkeys(self.non_terminal_states, 0)

    def getInitialActionValueFunctions(self) -> Mapping[S, Mapping[A, float]]:
        random_value = np.random.random()
        action_value = dict.fromkeys(self.actions, random_value)
        return dict.fromkeys(self.non_terminal_states, action_value)        

    def getStateValueFunctions(self) -> Mapping[S, float]:
        state_value_func = self.state_value_functions
        for s, data in self.policy_data:
            for action, prob in self.policy_data[s]:
                value = self.discount_factor * self.state_value_functions[s] * self.transition_data[s][action] + self.reward_data[s][action]
                state_value_func[s] = value
        return state_value_func

    def getActionValueFunctions(self) -> Mapping[S, Mapping[A, float]]:
        action_value_func = self.action_value_functions
        for s in self.states:
            for a in self.actions:
                value += self.discount_factor * self.reward_data[s][a] * self.policy_data[s][a]
                action_value_func[s] = value
        return action_value_func
        

if __name__ == '__main__':
    # transition is in the form of: Mapping[S, Mapping[A, Mapping[next_S, prob]]]
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
        3: {1: 1.0}
    }
    gamma = 0.8
    mdp = MDP(transition, reward, policy, gamma)
    