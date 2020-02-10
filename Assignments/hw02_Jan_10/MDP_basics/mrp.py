from typing import TypeVar, Sequence, Mapping, Set
import numpy as np
from scipy.linalg import eig
from mp import MP

S = TypeVar('S')
# R = TypeVar('R')

"""
Markov Reward Process is a Markov Chain with values
MRP is a tuple <S, P, R, gamma> where:
S is a finite set of states
P is a state transition probability matrix
R is a reward function: Rs = E[R_{t+1} | S_t = S]
gamma is a discount factor
"""

class MRP(MP):
    def __init__(self, transition_data: Mapping[S, Mapping[S, float]], gamma: float):
        self.reward_function = {}
        self.transition_data : Mapping[S, Mapping[S, float]] = transition_data
        self.states : Sequence[S] = self.getAllStates()   # inherited from MP
        self.terminal_states = self.getTerminalStates()
        self.non_terminal_states = self.getNonterminalStates()
        self.transition_matrix : np.ndarray = self.getTransitionMatrix()   # specific to MRP
        self.gamma : float = gamma
        self.reward_vector = self.getRewardVector()

    def getTerminalStates(self) -> Set[S]:
        terminal_states = []
        for s, trans_data in self.transition_data.items():
            if len(trans_data) == 1 and s in trans_data.keys():
                terminal_states.append(s)
        return terminal_states

    def getNonterminalStates(self) -> Set[S]:
        return [s for s in self.states if s not in self.terminal_states]
    
    # r(s, s') definition
    def setRewardFunctionDeterministic(self, reward_function: Mapping[S, float]) -> None:
        self.reward_function = reward_function
 
    # R(s) = \sum_{s'} p(s,s') * r(s,s') definition
    def setRewardFunctionUndeterministic(self, 
                reward_function_undeterministic: Mapping[S, Mapping[S, float]]) -> None:
        self.reward_function: Mapping[S, float] = dict.fromkeys(self.states, 0)
        for s, undeterministic_rewards in reward_function_undeterministic.items():
            for next_state, r in undeterministic_rewards.items():
                if next_state in self.transition_data[s].keys():
                    self.reward_function[s] += self.transition_data[s][next_state] * r

    # get reward vector after reward function is set by one of the two definitions
    def getRewardVector(self) -> np.ndarray :
        return np.array([self.reward_function.get(s, 0) for s in self.non_terminal_states])

    # get transitional matrix for non-terminal states
    def getTransitionMatrix(self) -> np.ndarray :
        # initialize transition matrix dimensions
        transition_matrix_dimension = len(self.non_terminal_states)
        transition_matrix = np.zeros((transition_matrix_dimension, 
                                    transition_matrix_dimension))
        # extract transition data and put into transition matrix
        for i, s_i in enumerate(self.non_terminal_states):
            for j, s_j in enumerate(self.non_terminal_states):
                transition_matrix[i, j] = self.transition_data[s_i].get(s_j, 0.)
        return transition_matrix

    # MRP value function based on Matrix Inversion method
    def getValueFunction(self) -> Mapping[S, float] :
        value_function = {}
        value_function_vector = np.linalg.inv(
            np.eye(len(self.non_terminal_states)) - self.gamma * self.transition_matrix).dot(self.getRewardVector())
        for idx, state in enumerate(self.non_terminal_states):
            value_function[state] = value_function_vector[idx]
        return value_function

if __name__ == "__main__":
    transitions = {
        1: {1: 0.1, 2: 0.8, 3: 0.1},
        2: {1: 0.4, 2: 0.2, 3: 0.4},
        3: {3: 1.0}
    }
    rewards = {1: 5.0, 2: 2.0, 3: 1.0}

    mrp = MRP(transitions, 1)
    mrp.setRewardFunctionDeterministic(rewards)

    print("Non-terminal state(s) are: ", mrp.non_terminal_states)
    print("Terminal state(s) are: ", mrp.terminal_states)
    print("Rewards function is: ", mrp.reward_function)
    print("Rewards vector is: ", mrp.getRewardVector())
    print("Value functions for states are: ", mrp.getValueFunction())