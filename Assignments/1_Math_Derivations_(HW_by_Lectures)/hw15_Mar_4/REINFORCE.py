from typing import TypeVar, Sequence, Mapping, Set, Tuple
import numpy as np

# Episodes should be a Sequence of Tuples
class REINFORCE:
    def __init__(self,
                 n : int,   # size of theta vector 
                 actions : Sequence,    # list of possible actions
                 episodes : Sequence[Tuple], # list of (state, action, reward)
                 theta : Sequence,    # n-vector
                 alpha : float, 
                 gamma : float, # discount factor
                 ):
        self.n = n
        self.actions = actions
        self.episodes = episodes
        self.theta = theta
        self.alpha = alpha
        self.gamma = gamma

    # scoring function based on softmax case derivation
    def scoreFunction(self, episode : Tuple):
        features_vector = np.random.random_sample((self.n, len(self.actions)))
        total_value = 0
        for action in range(len(self.actions)):
            total_value += features_vector[episode[0]][action]
        expected_value = total_value / len(self.actions)
        return features_vector[episode[0]][episode[1]] - expected_value


    def reinforceUpdate(self):
        for each_episode in self.episodes:
            T = len(self.episodes)
            for t in range(T):
                # G update
                for k in range(t, T):
                    G += self.gamma ** (k-t) * self.episodes[k][2]
                # theta update
                self.theta = self.theta + self.alpha * self.gamma ** t * self.scoreFunction(each_episode) * G