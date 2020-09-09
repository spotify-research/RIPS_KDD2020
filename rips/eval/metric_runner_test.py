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
import apache_beam as beam
import pytest
from apache_beam.runners.runner import PipelineState

from rips.eval.offpolicy.ir_based import PrecisionEstimator, RBPEstimator
from rips.eval.metric_runner import BeamListwiseMetricRunner
from rips.eval.metric_runner import ListwiseMetricRunner
from rips.policy.listwise_log import ListwiseLog
from rips.policy.logging_policy import DeterministicLoggingPolicy


def test_ListwiseMetricRunner():
    synthetic_logs = [(ListwiseLog("q1", ["d1", "d2", "d3"], [1, 0, 1]), {"test": {"d1": 0.9, "d2": 0.1, "d3": 0.4}})]

    estimators = [PrecisionEstimator(cut_offs=[1, 2, 3, 10]), RBPEstimator(cut_offs=[1, 2, 3, 10])]
    target_policies = [("test", DeterministicLoggingPolicy())]
    runner = ListwiseMetricRunner(estimators, target_policies, max_cutoff=10)
    estimates = runner.get_estimate(synthetic_logs)

    # Check precision results
    assert list(estimates["test"][estimators[0].name]["est_mean"]) == pytest.approx([1.0, 1.0, 0.66666666, 0.2])

    assert list(estimates["test"][estimators[1].name]["est_mean"]) == pytest.approx([0.5, 0.75, 0.75, 0.75])


def test_BeamListwiseMetricRunner():
    synthetic_logs = [(ListwiseLog("q1", ["d1", "d2", "d3"], [1, 0, 1]), {"test": {"d1": 0.9, "d2": 0.1, "d3": 0.4}})]
    cutoffs = [1, 2, 3, 10]
    estimators = [PrecisionEstimator(cut_offs=cutoffs), RBPEstimator(cut_offs=[1, 2, 3, 10])]

    def check(estimates):
        assert list(estimates["test"][estimators[0].name]["est_mean"]) == pytest.approx([1.0, 1.0, 0.66666666, 0.2])

        assert list(estimates["test"][estimators[1].name]["est_mean"]) == pytest.approx([0.5, 0.75, 0.75, 0.75])

    init_estimators_fn = lambda: estimators  # noqa:E731

    target_policies = [("test", DeterministicLoggingPolicy())]
    init_target_policy_fn = lambda: target_policies  # noqa:E731

    p = beam.Pipeline()
    (
        p
        | beam.Create(synthetic_logs)
        | BeamListwiseMetricRunner(init_estimators_fn, init_target_policy_fn, max(cutoffs))
        | beam.Map(check)
    )

    res = p.run()
    res.wait_until_finish()
    assert res.state == PipelineState.DONE
