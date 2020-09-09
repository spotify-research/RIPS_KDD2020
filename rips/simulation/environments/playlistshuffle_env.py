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
from typing import Dict, List, Tuple, Any

import numpy as np

from rips.policy import ListwiseLog
from rips.policy import Policy
from rips.simulation.environments import Environment


class PlaylistShuffleEnvironment(Environment):
    def get_interactions(self, context, rl):
        rewards = self.true_reward.reward_matrix()[context]
        labels = [rewards[c] for c in rl]

        user_interactions = self.user_simulator.get_rewards(rl, labels)
        if len(user_interactions) < len(rl):
            interactions = user_interactions + [None] * (len(rl) - len(user_interactions))
        else:
            interactions = user_interactions
        return interactions

    def _next_log(self, context=0):
        reward_mat = self.true_reward.reward_matrix()[context]
        candidates = np.squeeze(np.argwhere(reward_mat >= 0))
        true_rewards = reward_mat[candidates]
        display_rl, propensities = self.logging_policy.act(ListwiseLog(context, candidates, true_rewards))

        display_rl.extend(np.setdiff1d(candidates, display_rl))
        propensities = propensities + [None] * (len(display_rl) - len(propensities))

        interactions = self.get_interactions(context, display_rl)

        return ListwiseLog(context, display_rl, interactions, propensities)


##########################################################################################
# Below are some simulated logging policies
##########################################################################################


class SimulatedPlaylistTargetPolicy(Policy):
    def __init__(self, start_pos=0, step=1):
        super().__init__()
        self.start_pos = start_pos
        self.step = step

    def propensity(self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any):
        raise NotImplementedError

    def predictions(self, N):
        if self.step > 0:
            return {
                _id % N: float(1.0 / (rank + 1)) for rank, _id in enumerate(range(self.start_pos, self.start_pos + N))
            }
        else:
            return {
                _id % N: float(1.0 / (rank + 1))
                for rank, _id in enumerate(range(self.start_pos, self.start_pos - N, self.step))
            }

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        N = len(log.all_candidate_ids)
        ranking = self.predictions(N)
        rl = list(zip(*sorted(ranking.items(), key=lambda x: x[1], reverse=True)))[0]
        return list(rl), [1.0] * len(rl)
