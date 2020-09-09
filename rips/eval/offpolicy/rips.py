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
from __future__ import absolute_import, division

import logging

import numpy as np
from scipy.special import logsumexp

from rips.eval.offpolicy.base import OffpolicyEstimator


class RIPSEstimator(OffpolicyEstimator):
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

        N, L = weight_vector.shape
        logW = np.zeros((N, L))
        prev_W = np.zeros(N)
        W = np.zeros((N, L))

        for l in range(L):
            for n in range(N):
                logW[n, l] = prev_W[n] + np.log(weight_vector[n, l])
            W[:, l] = np.exp(np.log(N) + logW[:, l] - logsumexp(logW[:, l]))
            prev_W = np.log(W[:, l])

        logging.info("{} Weights Sum: {}".format(self.name, W.sum(axis=0)))
        scores = np.multiply(W, reward_vector)
        scores = scores.cumsum(axis=1)
        cut_off_est = self.index_cutoffs(scores)

        val_mean, val_rstd = self.get_mean_variance(cut_off_est)

        return val_mean, val_rstd
