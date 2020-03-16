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
        # transition data:     Mapping[S, Mapping[A, Mapping[next_S, prob]]], also a transition matrix
        self.transition_data : Mapping[S, Mapping[A, Mapping[S, float]]] = transition_data
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
        self.terminal_states = self.getTerminalStates()
        self.non_terminal_states = self.getNonterminalStates()
        self.gamma : float = gamma

    def getTerminalStates(self) -> Sequence[S]:
        terminal_states = []
        for s, trans_data in self.transition_data.items():
            if len(trans_data) == 1 and s in trans_data.keys():
                terminal_states.append(s)
        return terminal_states

    def getNonterminalStates(self) -> Sequence[S]:
        return list(set(self.states) - set(self.getTerminalStates()))

    def getActions(self) -> Sequence[A]:
        actions = []
        for data in self.transition_data.items():
            for a in data[1].keys():
                actions.append(a)
        return list(set(actions))

    def getActionsForState(self) -> Mapping[S, Sequence[A]]:
        actions_for_state = {}
        for state, data in self.transition_data.items():
            actions_for_state[state] = list(data.keys())
        return actions_for_state

    def getPolicyForState(self) -> Mapping[S, A]:
        policy_for_state = {}
        for s, action_with_prob in self.policy_data.items():
            highest_prob_action = np.random.choice(self.actions, 1, p=list(action_with_prob.values()))
            policy_for_state[s] = highest_prob_action
        return policy_for_state

    def toMRP(self) -> MRP:
        transition_data_mrp = {}
        rewards_data_mrp = {}
        for s, _ in self.transition_data.items():
            for _, action in self.getPolicyForState().items():
                transition_data_mrp[s] = self.transition_data[s][action[0]]

        for s, _ in self.reward_data.items():
            for _, action in self.getPolicyForState().items():
                rewards_data_mrp[s] = self.reward_data[s][action[0]]

        mrp = MRP(transition_data_mrp, self.discount_factor)
        mrp.setRewardFunctionDeterministic(rewards_data_mrp)
        return mrp        

    def getInitialStateValueFunctions(self) -> Mapping[S, float]:
        return dict.fromkeys(self.getNonterminalStates(), 0)

    def getInitialActionValueFunctions(self) -> Mapping[S, Mapping[A, float]]:
        random_value = np.random.random()
        action_value = dict.fromkeys(self.actions, random_value)
        return dict.fromkeys(self.getNonterminalStates(), action_value)        

    def getStateValueFunctions(self) -> Mapping[S, float]:
        # state_value_func = self.state_value_functions
        state_value_func = dict.fromkeys(self.states, 0)
        tmp_value = 0
        for s, _ in self.policy_data.items():
            for action, prob in self.policy_data[s].items():
                for next_state in self.states:
                    if next_state in self.transition_data[s][action].keys():
                        tmp_value += self.state_value_functions[next_state] * self.transition_data[s][action][next_state]
                if action in self.reward_data[s].keys():
                    tmp_value = tmp_value * self.discount_factor + self.reward_data[s][action]
                state_value_func[s] += tmp_value * prob
        return state_value_func

    def getActionValueFunctions(self) -> Mapping[S, Mapping[A, float]]:
        action_value_func = dict.fromkeys(self.states, dict.fromkeys(self.actions, 0))
        tmp_value = 0
        for s in self.states:
            for a in self.actions:
                for next_state in self.states:
                    for next_action in self.actions:
                        if next_action in self.policy_data[next_state] and self.action_value_functions[next_state].keys():
                            tmp_value += self.policy_data[next_state][next_action] * self.action_value_functions[next_state][next_action]
                    if next_state in self.transition_data[s][a].keys():
                        action_value_func[s][a] += tmp_value * self.transition_data[s][a][next_state]
                action_value_func[s][a] = self.discount_factor * action_value_func[s][a] + self.reward_data[s][a]
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
        1: {1: 5.0, 2: 15.0},
        2: {1: 3.0, 2: -3.0},
        3: {1: 0.0, 2: 0.0}
    }
    policy = {
        1: {1: 0.5, 2: 0.5},
        2: {1: 0.2, 2: 0.8},
        3: {1: 1.0, 2: 0.0}
    }
    gamma = 0.8
    mdp = MDP(transition, reward, policy, gamma)
    print("MDP state-value function is: ", mdp.getStateValueFunctions())
    print("MDP action-value function is: ", mdp.getActionValueFunctions())
    # mrp = mdp.toMRP()
    # print("MRP value function is: ", mrp.getValueFunction())
    