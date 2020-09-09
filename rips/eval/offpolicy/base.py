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
from typing import Union

import numpy as np

from rips.policy.listwise_log import ListwiseLog


class OffpolicyEstimator(ABC):
    def __init__(self, config: dict = None, cut_offs: Union[int, list] = None):
        """

        :param logging_policy:
        :param target_policy:
        :param config:
        :param cut_offs:
        """
        self.config = config or {}
        if cut_offs is None:
            self.cut_offs = [10]
        else:
            self.cut_offs = cut_offs

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def compute_score(self, log, target_rl, target_propensities):
        raise NotImplementedError

    @abstractmethod
    def get_estimate(self, score_tuples):
        raise NotImplementedError

    @staticmethod
    def extract_weights_reward(tuples):
        w_mat = np.asarray([w[0] for w in tuples])
        rewards = np.asarray([w[1] for w in tuples])
        return w_mat, rewards

    def index_cutoffs(self, scores):
        index = np.asarray(self.cut_offs) - 1
        return scores[:, index]

    def get_mean_variance(self, cut_off_est):
        val_mean = np.nanmean(cut_off_est, axis=0)
        val_rstd = np.std(cut_off_est, axis=0) / np.sqrt(cut_off_est.shape[0])
        return val_mean, val_rstd

    def get_propensity(self, log: ListwiseLog, rank: int):
        cand = log.candidates[rank]
        if cand is None:
            return 0.0
        else:
            assert cand.propensity is not None, "Propensity is not found in the log."
            return cand.propensity
