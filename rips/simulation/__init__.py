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
"""
    This package provides a sets of classes and functions
    to simulate user behavior.
"""
# flake8: noqa
from rips.simulation.environments import Environment
from rips.simulation.environments.playlistshuffle_env import PlaylistShuffleEnvironment
from rips.simulation.environments.playlistshuffle_env import SimulatedPlaylistTargetPolicy

from rips.simulation.reward_simulation import TrueRewardSimulator
from rips.simulation.reward_simulation import BetaDistributionRewards

from rips.simulation.user_simulation import UserSimulator
from rips.simulation.user_simulation import NoNoiseUserSimulator
from rips.simulation.user_simulation import ImpatientNoNoiseUserSimulator
from rips.simulation.user_simulation import HistoryAwareNoNoiseUserSimulator
from rips.simulation.user_simulation import CascadeHistoryAwareNoNoiseUserSimulator
