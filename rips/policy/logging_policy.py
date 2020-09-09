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
from abc import ABC
from typing import Dict, List, Tuple, Any

import numpy as np

from rips.policy import ListwiseLog
from rips.policy import Policy
from rips.policy import PredictionObjPolicy


class LoggingPolicy(ABC, Policy):
    def __init__(self, base_policy: Policy = PredictionObjPolicy(), config=None):
        super().__init__(config)
        self._with_replacement = self.config.get("with_replacement", False)
        self._base_policy = base_policy

    @property
    def with_replacement(self):
        return self._with_replacement

    def base_ranking(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        return self._base_policy.act(log, predictions, max_cutoff)

    def compute_propensities(self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None):
        """
            Convenience function to be used for off-policy evaluation
            The function pre-computes propensities for a given log line and returns a list.

        :param log:
        :param predictions:
        :param max_cutoff:
        :return:
        """
        base_ranking, _ = self.base_ranking(log, predictions, None)
        if max_cutoff is None:
            _max_cutoff = len(base_ranking)
        else:
            _max_cutoff = max_cutoff

        logged_rl = log.slate_ids
        max_cutoff = min(_max_cutoff, len(logged_rl))
        available_candidates = list(log.all_candidate_ids)
        propensities = []
        for rank in range(max_cutoff):
            candidate_id = logged_rl[rank]
            history = logged_rl[:rank]

            propensity = self.propensity(rank, history, available_candidates, (base_ranking, predictions))

            if not self.with_replacement:
                try:
                    available_candidates.remove(candidate_id)
                except ValueError:
                    pass  # This is to deal with duplicate entries

            propensities.append(propensity[candidate_id])
        return base_ranking[:_max_cutoff], propensities


class DeterministicLoggingPolicy(LoggingPolicy):
    def propensity(
        self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any
    ) -> Dict[str, float]:
        base_ranking, _ = policy_data
        return {_id: 1.0 if base_ranking[rank] == _id else 0.0 for _id in available_candidates}

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        rl, _ = self.base_ranking(log, predictions, None)
        return list(rl), [1.0] * len(rl)


class UniformLoggingPolicy(LoggingPolicy):
    def propensity(
        self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any
    ) -> Dict[str, float]:
        base_ranking, _ = policy_data
        return {_id: 1.0 / len(available_candidates) for _id in available_candidates}

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        base_ranking, _ = self.base_ranking(log, predictions, None)
        num_actions = len(base_ranking)
        if max_cutoff is None:
            max_cutoff = num_actions

        if self.with_replacement:
            rl = np.random.choice(base_ranking, size=max_cutoff, replace=True)
            propensities = [1.0 / float(num_actions) for _ in rl]
        else:
            rl = np.random.choice(base_ranking, size=max_cutoff, replace=False)
            propensities = [1 / float(num_actions - rank) for rank, c in enumerate(rl)]
        return list(rl), list(propensities)


class EpsilonGreedyLoggingPolicy(LoggingPolicy):
    @property
    def epsilon(self):
        return self.config.get("epsilon", 0.1)

    def propensity(
        self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any
    ) -> Dict[str, float]:
        base_ranking, _ = policy_data
        best_action_index = -1
        for idx, _id in enumerate(base_ranking):
            if _id not in subaction_history:
                best_action_index = available_candidates.index(_id)
                break
        if best_action_index < 0:
            return {_id: 0.0 for _id in available_candidates}

        num_candidates = len(available_candidates)

        pr = (self.epsilon / float(num_candidates)) * np.ones(num_candidates)
        pr[best_action_index] += 1.0 - float(self.epsilon)

        return {_id: pr[idx] for idx, _id in enumerate(available_candidates)}

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:

        base_ranking, _ = self.base_ranking(log, predictions, None)
        if max_cutoff is None:
            max_cutoff = len(base_ranking)
        available_candidates = list(log.all_candidate_ids)
        final_rl = []
        final_propensities = []
        for rank in range(max_cutoff):

            history = final_rl[: rank + 1]
            propensity_dict = self.propensity(rank, history, available_candidates, (base_ranking, predictions))

            sampled_id = np.random.choice(list(propensity_dict.keys()), size=1, p=list(propensity_dict.values()))[0]
            if not self.with_replacement:
                available_candidates.remove(sampled_id)

            final_propensities.append(propensity_dict[sampled_id])
            final_rl.append(sampled_id)

        return list(final_rl), list(final_propensities)


class ThompsonSamplingLoggingPolicy(LoggingPolicy):
    # TODO: Recheck this implemenation and uncomment
    def propensity(
        self, rank: int, subaction_history: List[str], available_candidates: List[str], policy_data: Any
    ) -> Dict[str, float]:
        raise NotImplementedError

    def act(
        self, log: ListwiseLog, predictions: Dict[str, float] = None, max_cutoff: int = None
    ) -> Tuple[List[str], List[float]]:
        raise NotImplementedError
