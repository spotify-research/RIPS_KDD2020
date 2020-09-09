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
from collections import defaultdict
from typing import Dict, List, Tuple

import apache_beam as beam

from rips.eval.offpolicy.base import OffpolicyEstimator
from rips.policy.listwise_log import ListwiseLog
from rips.policy.logging_policy import LoggingPolicy
from rips.utils.beam_utils import GroupAll


class ListwiseMetricRunner(object):
    def __init__(self, estimators, target_policies, max_cutoff=None, parser=None):
        self._estimators = estimators
        self._target_policies = target_policies
        self._max_cutoff = max_cutoff
        if parser is None:
            self.parser = lambda x: x
        else:
            self.parser = parser

    @property
    def target_policies(self) -> List[Tuple[str, LoggingPolicy]]:
        assert self._target_policies is not None, "Estimators not initialized"
        return self._target_policies

    @property
    def estimators(self) -> Dict[str, OffpolicyEstimator]:
        assert self._estimators is not None, "Estimators not initialized"
        if isinstance(self._estimators, list):
            self._estimators = {e.name: e for e in self._estimators}
        return self._estimators

    def score_log(self, _log_and_preds) -> Dict[tuple, tuple]:
        log_and_preds = self.parser(_log_and_preds)
        if isinstance(log_and_preds, tuple):
            log, pred_obj = log_and_preds
        else:
            log, pred_obj = log_and_preds, None
        scores = {}

        assert isinstance(log, ListwiseLog), "log must of be type: ListwiseLog"

        for target_policy_name, target_policy in self.target_policies:

            target_rl, target_propensities = target_policy.compute_propensities(
                log, pred_obj[target_policy_name], self._max_cutoff
            )

            for estimator_name, estimator in self.estimators.items():
                scores[(target_policy_name, estimator_name)] = estimator.compute_score(
                    log, target_rl, target_propensities
                )
        return scores

    def get_estimate(self, all_logs_and_preds):
        all_scores = defaultdict(list)
        for log in all_logs_and_preds:
            for estimator_name, score in self.score_log(log).items():
                all_scores[estimator_name].append(score)

        final_scores = defaultdict(dict)
        for (target_policy_name, estimator_name), scores in all_scores.items():
            estimator = self.estimators.get(estimator_name)
            est = estimator.get_estimate(scores)
            if len(est) == 2:
                est_mean, est_rstd = est
                final_scores[target_policy_name][estimator_name] = {
                    "est_mean": list(est_mean),
                    "est_rstd": list(est_rstd),
                }
            else:
                est_mean, est_rstd, data = est
                final_scores[target_policy_name][estimator_name] = {
                    "est_mean": list(est_mean),
                    "est_rstd": list(est_rstd),
                    "data": data,
                }

        return final_scores


class BeamListwiseMetricRunner(beam.PTransform):
    _estimators_estimators = None

    def __init__(
        self, init_estimators_fn, init_target_policy_fn, max_cutoff, parser=None, *unused_args, **unused_kwargs
    ):
        super(self.__class__).__init__(*unused_args, **unused_kwargs)
        self.init_estimators_fn = init_estimators_fn
        self.init_target_policy_fn = init_target_policy_fn
        self.max_cutoff = max_cutoff
        if parser is None:
            self.parser = lambda x: x
        else:
            self.parser = parser

    class ComputeScore(beam.DoFn, ListwiseMetricRunner):
        def __init__(
            self, init_estimators_fn, init_target_policy_fn, max_cutoff, parser, *unused_args, **unused_kwargs
        ):
            super(self.__class__).__init__(*unused_args, **unused_kwargs)
            self.init_estimators_fn = init_estimators_fn
            self.init_target_policy_fn = init_target_policy_fn
            self.max_cutoff = max_cutoff
            self.parser = parser

        def start_bundle(self):
            self._estimators = self.init_estimators_fn()
            self._target_policies = self.init_target_policy_fn()
            self._max_cutoff = self.max_cutoff

        def process(self, element, *args, **kwargs):
            for estimator, score in self.score_log(element).items():
                yield estimator, score

    class GetEstimate(beam.DoFn, ListwiseMetricRunner):
        def __init__(self, init_estimators_fn, *unused_args, **unused_kwargs):
            super(self.__class__).__init__(*unused_args, **unused_kwargs)
            self.init_estimators_fn = init_estimators_fn

        def start_bundle(self):
            self._estimators = self.init_estimators_fn()

        def process(self, element, *args, **kwargs):
            (target_policy_name, estimator_name), scores = element
            estimator = self.estimators.get(estimator_name)
            est = estimator.get_estimate(scores)

            if len(est) == 2:
                est_mean, est_rstd = est
                yield {
                    "target_policy_name": target_policy_name,
                    "estimator": estimator_name,
                    "est_mean": list(est_mean),
                    "est_std": list(est_rstd),
                }

            else:
                est_mean, est_rstd, data = est
                yield {
                    "target_policy_name": target_policy_name,
                    "estimator": estimator_name,
                    "est_mean": list(est_mean),
                    "est_std": list(est_rstd),
                    "data": data,
                }

    @staticmethod
    def flatten_results(result_list):
        result_map = defaultdict(dict)
        for res in result_list:
            result_map[res["target_policy_name"]][res["estimator"]] = {
                "est_mean": res["est_mean"],
                "est_std": res["est_std"],
                "data": res.get("data", None),
            }
        return result_map

    def expand(self, logs_and_pred_recs):
        return (
            logs_and_pred_recs
            | "ComputeScore"
            >> beam.ParDo(
                self.ComputeScore(self.init_estimators_fn, self.init_target_policy_fn, self.max_cutoff, self.parser)
            )
            | "GroupByEstimatorKey" >> beam.GroupByKey()
            | "GetEstimate" >> beam.ParDo(self.GetEstimate(self.init_estimators_fn))
            | "GroupAllEstimatorResult" >> GroupAll()
            | "FlattenResultIntoSingleMap" >> beam.Map(self.flatten_results)
        )
