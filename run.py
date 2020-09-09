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
from rips.job import run_job
from rips.policy import EpsilonGreedyLoggingPolicy
from rips.policy import UniformLoggingPolicy
from rips.simulation import BetaDistributionRewards
from rips.simulation import SimulatedPlaylistTargetPolicy as TP

# Simulation Setup
NUM_TRIALS = 2
NUM_CONTEXTS = 10
NUM_LOGS = 5000
NUM_TRACKS = 10
DEPTH = 10  # Max Number of tracks the simulated user will consider
SLATE_LIMIT = 10
CUT_OFFS = [3]  # [1, 3, 5, 10]
DATAFLOW_ARGS = {
    "project": "{INSERT-PROJECT-NAME-HERE}",
    "temp_location": "gs://{INSERT-BUCKET-NAME-HERE}/dataflow/tmp/",
    "staging_location": "gs://{INSERT-BUCKET-NAME-HERE}/dataflow/staging",
    "worker_machine_type": "n1-standard-8",
    "autoscaling_algorithm": "THROUGHPUT_BASED",
    "setup_file": "./setup.py",
    "subnetwork": "https://www.googleapis.com/compute/v1/projects/xpn-master/regions/europe-west1/subnetworks/xpn-euw1",
    "region": "europe-west1",
    "service_account": "{INSERT-SERVICE-ACCOUNT-NAME-HERE}",
    "runner": "DataflowRunner",
}

policies = {
    "logging_policy": UniformLoggingPolicy(TP(0), config={"with_replacement": False}),
    "1-ideal": EpsilonGreedyLoggingPolicy(base_policy=TP(0), config={"with_replacement": False}),
    "2-rand": UniformLoggingPolicy(TP(0), config={"with_replacement": False}),
    "3-worst": EpsilonGreedyLoggingPolicy(TP(NUM_TRACKS - 1, step=-1), config={"with_replacement": False}),
}
true_rewards = BetaDistributionRewards(NUM_TRACKS, NUM_CONTEXTS, alpha=0.3, beta=1)


def main(output_loc, local=False):
    run_job(
        output_loc=output_loc,
        policies=policies,
        true_rewards=true_rewards,
        num_trials=NUM_TRIALS,
        num_contexts=NUM_CONTEXTS,
        num_logs=NUM_LOGS,
        slate_depth=DEPTH,
        cut_off=CUT_OFFS,
        dataflow_args=DATAFLOW_ARGS,
    )


if __name__ == "__main__":
    import logging
    import fire

    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
