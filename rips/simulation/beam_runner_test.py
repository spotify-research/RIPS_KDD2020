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
from apache_beam.runners.runner import PipelineState

from rips.simulation import PlaylistShuffleEnvironment
from rips.simulation import SimulatedPlaylistTargetPolicy
from rips.simulation.beam_runner import BeamRankerSimulator
from rips.simulation.reward_simulation import BetaDistributionRewards
from rips.simulation.user_simulation import NoNoiseUserSimulator


def test_BeamRankerSimulator():
    N = 10
    num_logs = 10

    def get_simulator():
        true_rewards = BetaDistributionRewards(N, alpha=3, beta=4)
        logging_policy = SimulatedPlaylistTargetPolicy()
        user_sim = NoNoiseUserSimulator(depth=10)

        env = PlaylistShuffleEnvironment(num_logs, 1, true_rewards, logging_policy, user_sim)

        env = {"logging_policy": env}
        return env

    def check(log_obj):
        simulator, log = log_obj
        assert len(log.slate_ids) == N

    p = beam.Pipeline()
    (p | BeamRankerSimulator(num_logs, get_simulator) | beam.Map(check))
    res = p.run()
    res.wait_until_finish()
    assert res.state == PipelineState.DONE
