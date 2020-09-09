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
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple, Any

import numpy as np

from rips.policy.listwise_log import ListwiseLog


# TODO: Extend this to support trainable policies?
#   Currently we support this using a prediction object
#   (i.e., we get the predictions from the trained policy ahead of time).
class Policy(object):
    __metaclass__ = ABCMeta

    def __init__(self, config=None):
        self.config = config if config is not None else {}

    @abstractmethod
    def propensity(
        self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        """
            Given a context, candidates and features (optional) as a ListwistLog
                act must return a ranked list and propensities
        :param log: ListwiseLog containing context, candidates and features
        :param predictions: optional prediction dictionary used by certain policies
        :param max_cutoff: total number of items requested (used for efficiency reasons)
        :return: ranked list and propensities
        """
        pass


class PredictionObjPolicy(Policy):
    """
    Use this for if the predictions come from a trained
     policy and provided using the prediction object.
    """

    @abstractmethod
    def propensity(self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any):
        raise Exception("UnImplemented")

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        assert predictions is not None, "Predictions are required for deterministic policy"
        ranked_list = {c.id: float(predictions.get(c.id, -np.inf)) for c in log.candidates}
        rl = []
        propensity = np.zeros(len(ranked_list))
        for idx, (item_id, prop) in enumerate(sorted(ranked_list.items(), key=lambda x: x[1], reverse=True)):
            if max_cutoff is not None and idx > max_cutoff:
                break
            rl.append(item_id)
            propensity[idx] = prop
        propensity = propensity / propensity.sum()
        return rl, propensity
