from typing import TypeVar, Sequence, Mapping
import numpy as np
from scipy.linalg import eig

S = TypeVar('S')

"""
Markov Process is a tuple <S, P> where S is a (finite) set of states
and P is a state transition probability matrix
"""
class MP:
    def __init__(self, transition_data: Mapping[S, Mapping[S, float]]):
        self.transition_data : Mapping[S, Mapping[S, float]] = transition_data
        self.states : Sequence[S] = self.getAllStates()
        self.transition_matrix : np.ndarray = self.getTransitionMatrix()

    def getAllStates(self) -> Sequence[S]:
        return list(self.transition_data.keys())

    def getTransitionMatrix(self) -> np.ndarray :
        # initialize transition matrix dimensions
        transition_matrix_dimension = len(self.states)
        transition_matrix = np.zeros((transition_matrix_dimension, 
                                    transition_matrix_dimension))
        # extract transition data and put into transition matrix
        for i, s_i in enumerate(self.states):
            for j, s_j in enumerate(self.states):
                transition_matrix[i, j] = self.transition_data[s_i].get(s_j, 0.)
        return transition_matrix

    def stationaryDistribution(self) -> Mapping[S, float]:
        eigen_values, eigen_vector = eig(self.transition_matrix.transpose())
        # find strictly positive eigen vector
        for i in range(len(eigen_values)):
            if (eigen_values[i] - 1.0) < 1e-8:
                eigen_vector_data = np.array(eigen_vector[:, i]).astype(float)
        stationary_distribution = eigen_vector_data / np.linalg.norm(eigen_vector_data)
        return {s: stationary_distribution[i] for i, s in enumerate(self.states)}


if __name__ == '__main__':
    trans = {
        1: {1: 0.5, 2: 0.1, 3: 0.4},
        2: {1: 0.2, 2: 0.5, 3: 0.3},
        3: {3: 1.0}
    }

    mp = MP(trans)

    print('The states are: ', mp.states, '\n')
    print('The stationary distribution is: ', mp.stationaryDistribution(), '\n')
