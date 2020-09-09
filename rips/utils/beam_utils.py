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
    Transformers for various ML tasks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import json
import random

import apache_beam as beam


class JsonCoder(object):
    """A JSON coder interpreting each line as a JSON string."""

    @staticmethod
    def encode(row):
        """ encode input row into json string"""
        return json.dumps(row).encode("utf-8")

    @staticmethod
    def decode(row):
        """ decode json string into a row dict="""
        return json.loads(row.decode("utf-8"))


class InnerJoinTwo(beam.PTransform):
    def flat_map_to_single_elements(self, x):
        key, value = x

        value_keys = value.keys()
        value_vals = value.values()

        for instance in itertools.product(*value_vals):
            vals = dict(zip(value_keys, instance))
            c = {}
            for k, v in vals.items():
                c.update(v)
            yield c

    def expand(self, pipeline):
        return pipeline | beam.CoGroupByKey() | beam.FlatMap(self.flat_map_to_single_elements)


class ApproxSampler(beam.PTransform):
    """
    when beam.combiners.Sample is too slow use this to sample records
        from a PCollection. The returned PCollection might not contain
        the exact number of elements but would be close to the request
        sample size.
    """

    def __init__(self, sample_size):
        super(ApproxSampler, self).__init__()
        self.sample_size = sample_size

    @staticmethod
    def sample_items(record, total_count, sample_size):
        """
            set a flag to indicate if the record was sampled or not.
        Args:
            record (Any): input record.
            total_count (int): total number of records in the PCollection.
            sample_size (int): requested sample size.

        Returns:
            Tuple[bool, Any]: a tuple with sample indicator and the record.

        """
        if random.random() < sample_size / float(total_count):
            return True, record
        else:
            return False, record

    def default_label(self):
        return "ApproxSampler_%d" % self.sample_size

    def expand(self, pipeline):
        count = pipeline | "countTotal" >> beam.combiners.Count.Globally()

        return (
            pipeline
            | "%s_sample" % self.default_label
            >> beam.Map(self.sample_items, beam.pvalue.AsSingleton(count), self.sample_size)
            | "%s_filterSample" % self.default_label >> beam.Filter(lambda x: x[0])
            | beam.Map(lambda x: x[1])
        )


class GroupAll(beam.PTransform):
    @staticmethod
    def flatten(item):
        _, v = item
        return v

    def expand(self, pipeline):
        return pipeline | beam.Map(lambda x: (0, x)) | beam.GroupByKey() | beam.Map(self.flatten)


def _name(name):
    return "seqips_beam_global_state_%s" % name


def set_global(name, value):
    globals()[_name(name)] = value


def get_global(name):
    return globals().get(_name(name), None)


def delete_global(name):
    if _name(name) in globals():
        del globals()[_name(name)]


def get_or_initialize_global(name, initializer):
    value = get_global(name)
    if value is not None:
        return value
    value = initializer()
    set_global(name, value)
    return value


class _RoundRobinKeyFn(beam.DoFn):
    def __init__(self, count):
        self.count = count

    def start_bundle(self):
        self.counter = random.randint(0, self.count - 1)

    def process(self, element, *args, **kwargs):
        self.counter += 1
        if self.counter >= self.count:
            self.counter -= self.count
        yield self.counter, element


class LimitBundles(beam.PTransform):
    def __init__(self, count):
        self.count = count

    def expand(self, input_):
        return input_ | beam.ParDo(_RoundRobinKeyFn(self.count)) | beam.GroupByKey() | beam.FlatMap(lambda kv: kv[1])
