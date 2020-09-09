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


class PIPSEstimator(OffpolicyEstimator):
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
            weight_vector[rank] = target_propensities[rank] / logging_propensities[rank]

        return list(weight_vector), reward_vector

    def get_estimate(self, score_tuples):
        weight_vector, reward_vector = self.extract_weights_reward(score_tuples)

        N, K = weight_vector.shape
        W = np.zeros((N, K))
        for n in range(N):
            for pos in range(K):
                if pos == 0:
                    W[n, pos] = weight_vector[n, pos]
                # if not first subaction then multiply by previous weight
                else:
                    W[n, pos] = weight_vector[n, pos - 1] * weight_vector[n, pos]

        norm_weights = np.nan_to_num(N * W / W.sum(axis=0))
        # print("PIPS", norm_weights)
        w_times_r = np.multiply(norm_weights, reward_vector)
        scores = np.cumsum(w_times_r, axis=1)

        logging.info("{} Weights Sum: {}".format(self.name, norm_weights.sum(axis=0)))
        cut_off_est = self.index_cutoffs(scores)

        val_mean, val_rstd = self.get_mean_variance(cut_off_est)

        return val_mean, val_rstd
