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
from rips.policy import EpsilonGreedyLoggingPolicy
from rips.policy import UniformLoggingPolicy
from rips.simulation import BetaDistributionRewards
from rips.simulation import NoNoiseUserSimulator
from rips.simulation import PlaylistShuffleEnvironment
from rips.simulation import SimulatedPlaylistTargetPolicy


def test_PlaylistShuffleEnvironment():
    N = 10
    num_logs = 10
    true_rewards = BetaDistributionRewards(N, alpha=3, beta=4)
    logging_policy = UniformLoggingPolicy(base_policy=SimulatedPlaylistTargetPolicy(N))
    user_sim = NoNoiseUserSimulator(depth=10)

    env = PlaylistShuffleEnvironment(num_logs, 1, true_rewards, logging_policy, user_sim)

    log = env.next_log()

    assert len(log.candidates) == 10
    assert log.candidates[0].propensity == 0.1


def test_PlaylistShuffleEnvironment_shorter_depth():
    N = 10
    num_logs = 5
    true_rewards = BetaDistributionRewards(N, alpha=3, beta=4)
    logging_policy = EpsilonGreedyLoggingPolicy(base_policy=SimulatedPlaylistTargetPolicy(N))
    user_sim = NoNoiseUserSimulator(depth=5)

    env = PlaylistShuffleEnvironment(num_logs, 1, true_rewards, logging_policy, user_sim)
    env.generate_contexts(num_uniq_contexts=1, count=num_logs)
    log = env.next_log()

    assert len(log.candidates) == 10
    assert log.candidates[0].propensity >= 0.01
    assert log.candidates[5].reward is None


def test_SimulatedPlaylistTargetPolicy():
    N = 10
    policy = SimulatedPlaylistTargetPolicy(5)
    preds = policy.predictions(N)
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[0][0] == 5

    policy = SimulatedPlaylistTargetPolicy(0)
    preds = policy.predictions(N)
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[0][0] == 0

    policy = SimulatedPlaylistTargetPolicy(N - 1, step=-1)
    preds = policy.predictions(N)
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[0][0] == 9
    assert len(preds) == 10

    policy = SimulatedPlaylistTargetPolicy(N - 2, step=-1)
    preds = policy.predictions(N)
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[0][0] == 8
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[1][0] == 7
    assert sorted(preds.items(), key=lambda x: x[1], reverse=True)[2][0] == 6
    assert len(preds) == 10
