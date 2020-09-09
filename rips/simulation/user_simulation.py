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
from typing import List

import numpy as np


class UserSimulator(object):
    """

    Abstract class for simulating user behavior on a ranked list.

    **Usage of Simulators:**

    .. highlight:: python
    .. code-block:: python

        ranked_list = ['item-1', 'item-2', 'item-3']
        true_rewards = [0.8, 0.3, 0.2]
        sim = ImpatientNoNoiseUserSimulator()
        sim.get_rewards(ranked_list, true_rewards)

    """

    @staticmethod
    def from_config(config):
        if config.get("type") == "rbp":
            prob_stop = config.get("prob_stop")
            return ImpatientNoNoiseUserSimulator(prob_stop)
        elif config.get("type") == "rel":
            return NoNoiseUserSimulator()
        else:
            raise Exception("Invalid User Simulator Types")

    def get_rewards(self, ranked_list: List[str], true_rewards: List[float]) -> List[float]:
        """
            get the user interactions on a ranked list according to the user model.
        :param ranked_list: list of items in the ranked list
        :param true_rewards: list of true rewards for items items in `ranked_list`.
        :return: a list of interactions on the ranked list
        """
        raise NotImplementedError


class ImpatientNoNoiseUserSimulator(UserSimulator):
    """
    According to this model, the user sequentially traverses a ranked list,
    considering each item at a time, and decides the reward by sampling from
    a bernoulli distribution where the `p=true reward of the item`. In addition,
    the user abandons the ranked list according to a geometric distribution with
    a given theta parameter.

    """

    def __init__(self, prob_stop=0.5, depth=None):
        """
        :param prob_stop: probability of stopping or abandoning the ranked list
        :param depth: max number of items the user will consider. `If `None`, the user will \
        consider the entire ranked list.
        """
        self.prob_stop = prob_stop
        self.depth = depth

    def get_rewards(self, rl: List[str], true_rewards: List[float]) -> List[float]:
        if self.depth is not None:
            rl = np.take(rl, range(0, self.depth))

        rewards = np.zeros(len(rl))
        for idx, (_, true_reward) in enumerate(zip(rl, true_rewards)):
            rewards[idx] = np.random.binomial(1, p=true_reward)
            if np.random.rand() > self.prob_stop:
                break
        return list(rewards)


class NoNoiseUserSimulator(UserSimulator):
    """
    According to this model, the user sequentially traverses a ranked list,
    considering each item at a time, and decides the reward by sampling from
    a bernoulli distribution where the `p=true reward of the item`.

    """

    def __init__(self, depth: int = None, threshold: float = None):
        """
        :param depth: max number of items the user will consider. `If `None`, the user will \
        consider the entire ranked list.
        :param threshold: when threshold is set, if the true reward is above the threshold, \
        then the item gets a positive reward else negative.
        """
        self.depth = depth
        self.threshold = threshold

    def get_rewards(self, rl: List[str], true_rewards: List[float]) -> List[float]:
        if self.depth is not None:
            rl = np.take(rl, range(0, self.depth))

        rewards = []
        for _, true_reward in zip(rl, true_rewards):
            if self.threshold is None:
                rewards.append(np.random.binomial(1, p=true_reward))
            else:
                rewards.append(1 if true_reward > self.threshold else 0)
        return rewards


class HistoryAwareNoNoiseUserSimulator(UserSimulator):
    """
    History Aware User simulator that introduces reward interactions when a
    user is interacting with a ranked list. According to this model,
    the user sequentially traverses a ranked list, considering each item at a time,
    and decides the reward by sampling from a bernoulli distribution where the
    `p=true reward of the item`. In addition, when `rank > 1`, the true reward
    is multiplied by a factor is the prior reward is negative.

    """

    def __init__(self, prior_reward_effect: int = 2, depth: int = None):
        """
        :param prior_reward_effect: Factor by which the prior reward affect the current reward
        :param depth: max number of items the user will consider. \
        `If `None`, the user will consider the entire ranked list.
        """
        self.depth = depth
        self.prior_reward_effect = prior_reward_effect

    def get_rewards(self, rl: List[str], true_rewards: List[float]) -> List[float]:
        if self.depth is not None:
            rl = np.take(rl, range(0, self.depth))

        rewards = []
        for rank, (_, true_reward) in enumerate(zip(rl, true_rewards)):
            if rank > 0 and rewards[rank - 1] == 0:
                rewards.append(np.random.binomial(1, p=true_reward / self.prior_reward_effect))
            else:
                rewards.append(np.random.binomial(1, p=true_reward))
        return rewards


class CascadeHistoryAwareNoNoiseUserSimulator(UserSimulator):
    """
    Cascade History Aware User simulator that introduces reward interactions when a
    user is interacting with a ranked list. According to this model,
    the user sequentially traverses a ranked list, considering each item at a time,
    and decides the reward by sampling from a bernoulli distribution where the
    `p=true reward of the item`. When `rank > 1`, and the true reward for `rank - 1` is
    negative, then, all subsequent items get a negative reward.

    """

    def __init__(self, depth: int = None):
        """
        :param depth: max number of items the user will consider. \
        `If `None`, the user will consider the entire ranked list.
        """
        self.depth = depth

    def get_rewards(self, rl: List[str], true_rewards: List[float]) -> List[float]:
        if self.depth is not None:
            rl = np.take(rl, range(0, self.depth))

        rewards = []
        for rank, (_, true_reward) in enumerate(zip(rl, true_rewards)):
            if rank > 0 and rewards[rank - 1] > 0:
                rewards.append(np.random.binomial(1, p=1.0))
            else:
                rewards.append(np.random.binomial(1, p=true_reward))
        return rewards
