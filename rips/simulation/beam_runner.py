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


class BeamRankerSimulator(beam.PTransform):
    def __init__(self, num_logs, sim_init_fn, *unused_args, **unused_kwargs):
        super(self.__class__).__init__(*unused_args, **unused_kwargs)
        self.num_logs = num_logs
        self.sim_init_fn = sim_init_fn

    class SimulateLogs(beam.DoFn):
        simulator = None

        def __init__(self, sim_init_fn, *unused_args, **unused_kwargs):
            super(self.__class__).__init__(*unused_args, **unused_kwargs)
            self.sim_init_fn = sim_init_fn

        def start_bundle(self):
            self.simulators = self.sim_init_fn()

        def process(self, element_id, *args, **kwargs):
            for simulator_name, simulator in self.simulators.items():
                log = simulator.next_log()
                log.log_id = element_id
                yield simulator_name, log

    def expand(self, pipeline):
        return (
            pipeline
            | "CreateLogCounters" >> beam.Create(range(self.num_logs))
            | "SimulateLog" >> beam.ParDo(self.SimulateLogs(self.sim_init_fn))
        )
