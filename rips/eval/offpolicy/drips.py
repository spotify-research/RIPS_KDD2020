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


class DynamicRIPSEstimator(OffpolicyEstimator):
    @property
    def ess_threshold(self):
        return self.config.get("ess_threshold", 0.0001)

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
        def ESS(ws):
            return 1.0 / ((ws ** 2).sum())

        weight_vector, reward_vector = self.extract_weights_reward(score_tuples)

        N, L = weight_vector.shape
        W = np.zeros((N, L))
        from collections import defaultdict

        ess_logger = defaultdict(list)
        for l in range(L):
            ws = np.ones(N)
            s = 0  # lookback size
            prev_ess = np.inf
            while s < (l + 1):
                log_ws = np.log(ws) + np.log(weight_vector[:, l - s])
                ws = np.exp(np.log(N) + log_ws - logsumexp(log_ws))
                # print('position, ESS, lookback: ', l, ESS(ws / float(N)), s)

                ess = ESS(ws / float(N))
                ess_logger[l].append({"ess": ess, "lookback": s})
                bad_ess = ((ess / float(N)) < self.ess_threshold) or (ess > prev_ess)  # ess should not go up
                if s > 0 and bad_ess:
                    break
                W[:, l] = ws  # assign working value after test above fails
                prev_ess = ess
                s += 1

        logging.info("{} Weights Sum: {}".format(self.name, W.sum(axis=0)))
        scores = np.multiply(W, reward_vector)
        scores = scores.cumsum(axis=1)
        cut_off_est = self.index_cutoffs(scores)

        val_mean, val_rstd = self.get_mean_variance(cut_off_est)

        return val_mean, val_rstd, ess_logger
