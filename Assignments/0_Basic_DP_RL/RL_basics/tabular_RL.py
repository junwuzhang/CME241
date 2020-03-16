from typing import TypeVar, Sequence, Mapping, Set, Tuple
import numpy as np
import random

S = TypeVar('S')
A = TypeVar('A')

class tabularRL():
    def __init__(self,
                 state_action_relation : Mapping[S, Sequence[A]],
                 state_action_nextstate_reward_relation : Mapping[S, Mapping[A, Mapping[S, Tuple[float, float]]]],
                 terminal_states : Sequence[S], 
                 gamma : float):
        self.state_action_relation = state_action_relation
        self.state_action_nextstate_reward_relation = state_action_nextstate_reward_relation
        self.terminal_states = terminal_states
        self.gamma = gamma
        
    def generateState(self) -> S:
        return [state for state in self.state_action_relation.keys()][random.randint(0, len(self.state_action_relation.keys()) - 1)]

    def generateStateActionPair(self) -> Tuple[S, A]:
        state = self.generateState()
        action = [a for a in self.state_action_relation[state]][random.randint(0, len(self.state_action_relation[state]) - 1)]
        return [self.generateState(), action]