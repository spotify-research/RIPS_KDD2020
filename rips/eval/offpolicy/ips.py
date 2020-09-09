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
import logging

import numpy as np

from rips.eval.offpolicy.base import OffpolicyEstimator


class IPSEstimator(OffpolicyEstimator):
    def compute_score(self, log, target_rl, target_propensities):
        max_cutoff = np.max(self.cut_offs)

        reward_vector = log.slate_rewards[:max_cutoff]
        reward_vector = reward_vector + [0.0] * (max_cutoff - len(reward_vector))
        target_propensities = np.asarray(target_propensities[:max_cutoff])

        weight_vector = np.zeros(max_cutoff)
        logging_propensities = np.zeros(max_cutoff)

        for rank, candidate_id in enumerate(log.slate_ids):
            if rank >= max_cutoff or not log.is_exposed(candidate_id):
                break

            logging_propensities[rank] = self.get_propensity(log, rank)
            weight_vector[rank] = target_propensities[: (rank + 1)].prod() / logging_propensities[: (rank + 1)].prod()
        #     print(rank, target_propensities[rank], logging_propensities[rank], weight_vector[rank])
        # print(weight_vector)

        return list(weight_vector), reward_vector

    def get_estimate(self, score_tuples):
        weight_vector, reward_vector = self.extract_weights_reward(score_tuples)

        rewards = np.cumsum(reward_vector, axis=1)
        scores = np.multiply(weight_vector, rewards)
        cut_off_est = self.index_cutoffs(scores)

        val_mean, val_rstd = self.get_mean_variance(cut_off_est)

        return val_mean, val_rstd


class NormIPSEstimator(IPSEstimator):
    def get_estimate(self, score_tuples):
        weight_vector, reward_vector = self.extract_weights_reward(score_tuples)

        N = weight_vector.shape[0]
        norm_weights = np.nan_to_num(N * weight_vector / weight_vector.sum(axis=0))
        rewards = np.cumsum(reward_vector, 1)

        scores = np.multiply(norm_weights, rewards)
        logging.info("{} Weights Sum: {}".format(self.name, norm_weights.sum(axis=0)))
        cut_off_est = self.index_cutoffs(scores)

        val_mean, val_rstd = self.get_mean_variance(cut_off_est)

        return val_mean, val_rstd
