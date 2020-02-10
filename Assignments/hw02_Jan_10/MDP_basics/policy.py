from typing import Mapping, Dict, TypeVar
import numpy as np
from scipy.linalg import eig
from mp import MP

S = TypeVar('S')
A = TypeVar('A')

"""
Policy is a distribution over actions given states
"""

class Policy:
    def __init__(self, pi: Dict[S, Mapping[A, float]]):
        self.pi : Dict[S, Mapping[A, float]] = pi

    def get_state_probabilities(self, state: S) -> Mapping[A, float]:
        return self.pi[state]

    def get_state_action_probability(self, state: S, action: A) -> float:
        return self.get_state_probabilities(state).get(action, 0.)