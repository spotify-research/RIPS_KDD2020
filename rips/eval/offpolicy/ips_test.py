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
import numpy as np
import pytest

from rips.eval.offpolicy.ips import IPSEstimator, NormIPSEstimator
from rips.policy.listwise_log import ListwiseLog


@pytest.fixture()
def target_ranking():
    return {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.2, "e": 0.1}


@pytest.fixture()
def target_ranking1():
    return {"a": 0.5, "b": 0.4, "c": 0.3, "d": 0.1, "e": 0.2}


@pytest.fixture()
def log():
    return ListwiseLog("301", ["a", "b", "c", "d", "e"], [0, 1, 1, 0, 1], propensities=[1.0] * 5)


def test_IPSEstimator(log, target_ranking):
    synth_log = [(log, target_ranking)] * 1

    estimator = IPSEstimator(cut_offs=[3, 5])
    scores = []
    for l, p in synth_log:
        target_rl, prop = zip(*sorted(p.items(), key=lambda x: x[1], reverse=True))
        prop = [1.0] * len(target_rl)  # adding dummy propensities
        scores.append(estimator.compute_score(l, target_rl, prop))

    est_mean, _ = estimator.get_estimate(scores)
    expected = [l.reward * 1.0 for r, l in enumerate(log.candidates)]
    expected = np.cumsum(expected)[4]
    assert est_mean[1] == pytest.approx(expected)


def test_IPSEstimator_wExpPolicy(log, target_ranking):
    synth_log = [(log, target_ranking)] * 1000
    estimator = IPSEstimator(cut_offs=[3, 5])

    scores = []
    for l, p in synth_log:
        target_rl, prop = zip(*sorted(p.items(), key=lambda x: x[1], reverse=True))
        prop = [1.0] * len(target_rl)  # adding dummy propensities
        scores.append(estimator.compute_score(l, target_rl, prop))
    estimator.get_estimate(scores)
    # TODO: figure out a test case


def test_NIPSEstimator(log, target_ranking1):
    synth_log = [(log, target_ranking1)] * 1

    estimator = NormIPSEstimator(cut_offs=[3, 5])
    scores = []
    for l, p in synth_log:
        target_rl, prop = zip(*sorted(p.items(), key=lambda x: x[1], reverse=True))
        prop = [1.0] * len(target_rl)  # adding dummy propensities
        scores.append(estimator.compute_score(l, target_rl, prop))
    est_mean, _ = estimator.get_estimate(scores)
    assert est_mean[1] == 3.0
