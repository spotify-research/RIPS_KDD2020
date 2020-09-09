#
# Copyright 2020 Spotify AB
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from abc import ABC, abstractmethod

import numpy as np


class TrueRewardSimulator(ABC):
    @abstractmethod
    def reward_matrix(self) -> np.ndarray:
        raise NotImplementedError

    def load(self):
        pass

    def save(self):
        pass


class BetaDistributionRewards(TrueRewardSimulator):
    """
    Simulating true reward distribution for a given number of items.
    The method uses beta distribution for assigning a reward probability
    for each item. The reward probabilities are sorted so they can be
    compared to another draw (useful to simulate rankers).


    For example, we can model the number of plays/skips for a track by a series of successes and failures
    with alpha and beta parameters representing our prior expectation.

    """

    def __init__(
        self, num_items: int, num_contexts: int = 1, alpha: float = 3, beta: float = 4, normalize: bool = False
    ):
        """
        :param num_items: number of candidate items in a ranked list.
        :param num_contexts: number of contexts to generate.
        :param alpha: alpha parameter for the beta distribution.
        :param beta: beta parameter for the beta distribution.
        :return: a probability distribution over the number of items
        """
        self.mat = []
        for i in range(num_contexts):
            mat = np.asarray([sorted(np.random.beta(alpha, beta, size=num_items))])
            if normalize:
                mat = mat / mat.sum(axis=1)
            self.mat.append(mat)
        self.mat = np.vstack(self.mat)

    def reward_matrix(self) -> np.ndarray:
        return self.mat
