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
import pytest

from rips.eval.offpolicy.ir_based import MRREstimator
from rips.eval.offpolicy.ir_based import NDCGEstimator
from rips.eval.offpolicy.ir_based import PrecisionEstimator
from rips.eval.offpolicy.ir_based import RBPEstimator
from rips.policy.listwise_log import ListwiseLog


@pytest.fixture()
def single_q_run():
    return {
        "CR93E-3103": 2.129133,
        "CR93E-1282": 1.724760,
        "CR93E-1850": 2.104024,
        "CR93E-2473": 1.912871,
        "CR93E-3284": 1.800881,
        "CR93E-10505": 1.870957,
        "CR93E-1952": 1.677013,
        "CR93E-2191": 1.999081,
        "CR93E-1860": 1.732111,
    }


@pytest.fixture()
def judgments():
    # Copied from trec_eval tests folders --> qrels.test
    return ListwiseLog(
        "301",
        [
            "CR93E-10279",
            "CR93E-10505",
            "CR93E-1282",
            "CR93E-1850",
            "CR93E-1860",
            "CR93E-1952",
            "CR93E-2191",
            "CR93E-2473",
            "CR93E-3103",
            "CR93E-3284",
        ],
        [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    )


def test_PrecisionEstimator_score(judgments, single_q_run):
    synth_log = (judgments, single_q_run)

    estimator = PrecisionEstimator(cut_offs=[5, 10])
    target_rl, scores = zip(*sorted(single_q_run.items(), key=lambda x: x[1], reverse=True))
    score = estimator.compute_score(synth_log[0], target_rl, None)
    assert score[0] is None
    assert score[1] == pytest.approx([0.2, 0.2])


def test_NDCGEstimator_score(judgments, single_q_run):
    synth_log = (judgments, single_q_run)

    estimator = NDCGEstimator(cut_offs=[5, 10])
    target_rl, scores = zip(*sorted(single_q_run.items(), key=lambda x: x[1], reverse=True))
    score = estimator.compute_score(synth_log[0], target_rl, None)
    assert score[0] is None
    assert score[1] == pytest.approx([0.6131, 0.8066], rel=0.002)


def test_RBPEstimator_score(judgments, single_q_run):
    synth_log = (judgments, single_q_run)
    estimator = RBPEstimator(cut_offs=[5, 10])
    target_rl, scores = zip(*sorted(single_q_run.items(), key=lambda x: x[1], reverse=True))
    score = estimator.compute_score(synth_log[0], target_rl, None)
    assert score[0] is None
    assert score[1] == pytest.approx([0.5, 0.5039], rel=0.002)


def test_MRREstimator(judgments, single_q_run):
    synth_log = (judgments, single_q_run)
    estimator = MRREstimator(cut_offs=[5, 10])
    target_rl, scores = zip(*sorted(single_q_run.items(), key=lambda x: x[1], reverse=True))
    score = estimator.compute_score(synth_log[0], target_rl, None)
    assert score[0] is None
    assert score[1] == pytest.approx([1.0])

    assert estimator.get_estimate([score])[0] == 1.0
