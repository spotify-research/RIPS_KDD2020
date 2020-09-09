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
from typing import Union

import numpy as np


class ListwiseCandidate(object):
    def __init__(self, _id: Union[str, int], _reward: float, _propensity: float, _is_exposed: bool = True):
        self.id = _id
        self.propensity = _propensity
        self.reward = _reward
        self.is_exposed = _is_exposed


class ListwiseLog(object):
    @classmethod
    def from_tf_sequence_examples(cls, seq_example):
        raise NotImplementedError

    def __init__(self, context, candidates, rewards=None, propensities=None, data=None, log_id=None):
        """

        :param context:
        :param candidates:
        :param rewards:
        :param data:
        """
        self.log_id = log_id

        self._exposed_candidates = set()
        self._unexposed_candidates = set()
        self._context = context
        self._data = data

        self._candidates = []
        self._reward_map = dict()
        if rewards is None:
            rewards = [None] * len(candidates)
        if propensities is None:
            propensities = [None] * len(candidates)

        for c, r, p in zip(candidates, rewards, propensities):
            if isinstance(c, ListwiseCandidate):
                candidate = c
            elif isinstance(c, dict):
                assert "id" in c, "Each candidate must have a id field"
                assert "propensity" in c, "Each candidate must have a propensity field"
                if "is_exposed" in c:
                    is_exposed = c["is_exposed"]
                else:
                    is_exposed = self._get_is_exposed(r)
                candidate = ListwiseCandidate(c["id"], r, c.get("propensity", 1.0), is_exposed)
            else:
                is_exposed = self._get_is_exposed(r)
                candidate = ListwiseCandidate(c, r, p, _is_exposed=is_exposed)

            self._candidates.append(candidate)
            if candidate.is_exposed:
                self._exposed_candidates.add(candidate.id)
                self._reward_map[candidate.id] = candidate.reward
            else:
                self._unexposed_candidates.add(candidate.id)

    @staticmethod
    def _get_is_exposed(r):
        if r is None or np.isnan(r):
            is_exposed = False
        else:
            is_exposed = True
        return is_exposed

    @property
    def context(self):
        return self._context

    @property
    def features(self):
        return self._data

    @property
    def candidates(self):
        return self._candidates

    @property
    def all_candidate_ids(self):
        return self._exposed_candidates.union(self._unexposed_candidates)

    @property
    def slate_ids(self):
        return [c.id for c in self._candidates if c.is_exposed]

    @property
    def slate_rewards(self):
        return [c.reward for c in self._candidates if c.is_exposed]

    def get_reward(self, _id, default=0):
        return self._reward_map.get(_id, default)

    def is_exposed(self, _id):
        return _id in self._exposed_candidates

    def __str__(self):
        return "\n".join(
            [
                "id:{}\treward:{}\tpropensity:{}\texposed:{}".format(c.id, c.reward, c.propensity, c.is_exposed)
                for c in self.candidates
            ]
        )
