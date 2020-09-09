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
from apache_beam.options.pipeline_options import PipelineOptions, SetupOptions
from os.path import join
from rips.eval.metric_runner import BeamListwiseMetricRunner
from rips.simulation.beam_runner import BeamRankerSimulator
from rips.utils.beam_utils import JsonCoder, GroupAll
from functools import partial
import numpy as np


def init_env_fn(num_logs, num_contexts, true_rewards, policies, depth):
    from rips.simulation import CascadeHistoryAwareNoNoiseUserSimulator
    from rips.simulation import PlaylistShuffleEnvironment

    user_sim = CascadeHistoryAwareNoNoiseUserSimulator(depth=depth)
    on_policy_envs = {}

    # Init Environments
    for policy_name, policy in policies.items():
        env = PlaylistShuffleEnvironment(num_logs, num_contexts, true_rewards, policy, user_sim)
        on_policy_envs[policy_name] = env
    return on_policy_envs


def init_estimators_fn(cutoffs):
    from rips.eval import IIPSEstimator
    from rips.eval import NormIPSEstimator
    from rips.eval import PIPSEstimator
    from rips.eval import RIPSEstimator

    estimators = [
        NormIPSEstimator(cut_offs=cutoffs),
        IIPSEstimator(cut_offs=cutoffs),
        PIPSEstimator(cut_offs=cutoffs),
        RIPSEstimator(cut_offs=cutoffs),
    ]
    return estimators


def addTargetPolicies(log, policy_names, policies):
    pred_obj = {}
    for policy_name in policy_names:
        if policy_name in policies:
            N = len(log.candidates)
            pred_obj[policy_name] = policies[policy_name]._base_policy.predictions(N)
        else:
            raise Exception("Invalid Policy Name: {}".format(policy_name))
    return log, pred_obj


def flatten_onpolicy_results(res, cutoffs):
    online_estimates = dict()
    for d in res:
        run_cutoff, score = d
        run = run_cutoff.split(":")[0]
        if run not in online_estimates:
            online_estimates[run] = np.zeros(len(cutoffs))
        cutoff_idx = cutoffs.index(int(run_cutoff.split(":")[1]))
        online_estimates[run][cutoff_idx] = score
    for run in online_estimates:
        online_estimates[run] = list(online_estimates[run])
    return online_estimates


def run_job(
    output_loc,
    policies,
    true_rewards,
    num_trials,
    num_contexts,
    num_logs,
    slate_depth,
    cut_off,
    dataflow_args,
):
    def init_target_policies_fn():
        target_policies = [(p[0], p[1]) for p in policies.items() if p[0] != "logging_policy"]
        return target_policies

    target_policy_names = list(zip(*init_target_policies_fn()))[0]

    pipeline_options = PipelineOptions(dataflow_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True
    pipeline = beam.Pipeline(options=pipeline_options)

    _init_env_fn = partial(
        init_env_fn,
        num_logs=num_logs,
        num_contexts=num_contexts,
        true_rewards=true_rewards,
        policies=policies,
        depth=slate_depth,
    )
    _init_estimators_fn = partial(init_estimators_fn, cutoffs=cut_off)
    _flatten_onpolicy_results = partial(flatten_onpolicy_results, cutoffs=cut_off)

    for trial in range(num_trials):
        logs = pipeline | "LogSimulation[T-{}]".format(trial) >> BeamRankerSimulator(num_logs, _init_env_fn)

        (
            logs
            | "FilterLoggingPolicyLog[T-{}]".format(trial) >> beam.Filter(lambda x: x[0] == "logging_policy")
            | "AddPredictions[T-{}]".format(trial)
            >> beam.Map(lambda x: addTargetPolicies(x[1], target_policy_names, policies=policies))
            | "ListwiseMetricRunner[T-{}]".format(trial)
            >> BeamListwiseMetricRunner(
                _init_estimators_fn,
                init_target_policies_fn,
                max_cutoff=max(cut_off),
            )
            | "WriteToFile[T-{}]".format(trial)
            >> beam.io.WriteToText(
                join(output_loc, "trial-{}-results".format(trial)),
                file_name_suffix=".json",
                coder=JsonCoder,
            )
        )
        (
            logs
            | "SumRewards[T-{}]".format(trial)
            >> beam.FlatMap(lambda l: [(l[0] + ":" + str(c), sum(l[1].slate_rewards[:c])) for c in cut_off])
            | "ComputeMean[T-{}]".format(trial) >> beam.transforms.combiners.Mean.PerKey()
            | "GroupAll[T-{}]".format(trial) >> GroupAll()
            | "FlattenResultIntoSingleMap[T-{}]".format(trial) >> beam.Map(_flatten_onpolicy_results)
            | "WriteToOnPolicyFile[T-{}]".format(trial)
            >> beam.io.WriteToText(
                join(output_loc, "trial-{}-onpolicy".format(trial)),
                file_name_suffix=".json",
                coder=JsonCoder,
            )
        )
    results = pipeline.run()
    results.wait_until_finish()
