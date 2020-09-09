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
from abc import abstractmethod

import numpy as np

from rips.policy.listwise_log import ListwiseLog
from rips.eval.offpolicy.base import OffpolicyEstimator
from rips.eval.metrics.position_based import rbp, dcg, ndcg, precision, mrr


class BaseIREstimator(OffpolicyEstimator):
    cut_offs = None
    metric = None

    @property
    def unseen_nonrel(self):
        return self.config.get("unseen_nonrel", True)

    @abstractmethod
    def _metric_fn(self, rewards):
        raise NotImplementedError

    def compute_score(self, log: ListwiseLog, target_rl, target_propensities):
        max_cutoff = np.max(self.cut_offs)

        def get_reward(_id):
            if log.is_exposed(_id):
                return log.get_reward(_id)
            elif self.unseen_nonrel:
                return 0.0
            else:
                return None

        rewards = [get_reward(_id) for _id in target_rl[:max_cutoff]]
        estimate = self._metric_fn(rewards)
        return None, list(estimate)

    def get_estimate(self, score_tuples):
        scores = np.asarray([s[1] for s in score_tuples])
        val_mean = np.nanmean(scores, axis=0)
        val_rstd = np.std(scores, axis=0) / np.sqrt(scores.shape[0])

        return val_mean, val_rstd


class RBPEstimator(BaseIREstimator):
    def _metric_fn(self, rewards):
        return rbp(rewards, self.cut_offs)


class DCGEstimator(BaseIREstimator):
    def _metric_fn(self, rewards):
        return dcg(rewards, self.cut_offs)


class NDCGEstimator(BaseIREstimator):
    def _metric_fn(self, rewards):
        ideal = sorted(rewards, reverse=True)
        return ndcg(rewards, ideal, self.cut_offs)


class PrecisionEstimator(BaseIREstimator):
    def _metric_fn(self, rewards):
        return precision(rewards, self.cut_offs)


class MRREstimator(BaseIREstimator):
    def _metric_fn(self, rewards):
        return [mrr(rewards)]

    def get_estimate(self, score_tuples):
        scores = np.asarray([s[1] for s in score_tuples])
        val_mean = np.nanmean(scores, axis=0)
        val_rstd = np.std(scores, axis=0) / np.sqrt(scores.shape[0])

        return val_mean, val_rstd
