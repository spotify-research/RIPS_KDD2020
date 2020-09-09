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
from rips.policy import ListwiseLog


def test_listwise_log_list():
    synthetic_log = {"context": "q1", "candidates": ["d1", "d2", "d3"], "rewards": [1, 0, 1]}
    log = ListwiseLog(**synthetic_log)
    assert len(log.candidates) == 3
    assert log.slate_rewards == [1.0, 0.0, 1.0]
    assert log.get_reward("d1") == 1
    assert log.get_reward("d2") == 0
    assert log.get_reward("d3") == 1
    assert log.context == "q1"


def test_listwise_log_dict():
    synthetic_log = {
        "context": "q1",
        "candidates": [{"id": "d1", "propensity": 1}, {"id": "d2", "propensity": 1}, {"id": "d3", "propensity": 1}],
        "rewards": [1, 0, 1],
    }
    log = ListwiseLog(**synthetic_log)
    assert len(log.candidates) == 3
    assert log.slate_rewards == [1.0, 0.0, 1.0]
    assert log.get_reward("d1") == 1
    assert log.get_reward("d2") == 0
    assert log.get_reward("d3") == 1
    assert log.context == "q1"
