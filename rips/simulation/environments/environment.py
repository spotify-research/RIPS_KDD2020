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
from enum import Enum

import numpy as np

from rips.policy import ListwiseLog
from rips.policy import Policy
from rips.simulation.reward_simulation import TrueRewardSimulator
from rips.simulation.user_simulation import UserSimulator


class ContextSimulator(Enum):
    UniformRandom = 1
    PowerLawDistribution = 1


# TODO: generalize this later, currently support only Listwise
class Environment(ABC):
    context_list = []

    def __init__(
        self,
        num_logs,
        num_uniq_contexts,
        true_reward: TrueRewardSimulator,
        logging_policy: Policy,
        user_simulator: UserSimulator,
    ):
        self.logging_policy = logging_policy
        self.user_simulator = user_simulator
        self.true_reward = true_reward
        self.generate_contexts(num_uniq_contexts=num_uniq_contexts, count=num_logs)

    def generate_contexts(
        self, num_uniq_contexts: int, count: int, simulator: ContextSimulator = ContextSimulator.UniformRandom
    ):
        if simulator == ContextSimulator.UniformRandom:
            self.context_list = list(np.random.randint(num_uniq_contexts, size=count))

    def next_context(self) -> int:
        try:
            return self.context_list.pop()
        except IndexError:
            raise Exception("Make sure generate_contexts() is " "called before generating logs.")

    def next_log(self) -> ListwiseLog:
        return self._next_log(self.next_context())

    @abstractmethod
    def _next_log(self, context=0) -> ListwiseLog:
        raise NotImplementedError

    @abstractmethod
    def get_interactions(self, context, rl):
        pass
