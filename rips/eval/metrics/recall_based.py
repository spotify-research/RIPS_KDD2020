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
""" Recall Based Ranking Metrics
        recall
        average precision
        r_precision
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def recall(judged_ranked_list, ideal_grades, ret_rels=None, tot_true_rels=None):
    """Computes recall for the given ranked list

    Args:
        judged_ranked_list (List[int]): judged ranked list (list of 1s & 0s).
        ideal_grades ([List[int]]): ideal ranked list (list of 1s & 0s).
        ret_rels ([int], optional): Defaults to None. total relevant retrieved.
        When None, its inferred from the judged_ranked_list
        tot_true_rels ([type], optional): Defaults to None. total true relevant
        for the ranked list. If None, its inferred from ideal_grades.

    Returns:
        [float]: recall score for the ranked list
    """

    if ret_rels is None:
        ret_rels = sum(judged_ranked_list[~np.isnan(judged_ranked_list)] > 0)
    if tot_true_rels is None:
        tot_true_rels = sum(ideal_grades > 0)
    recall_score = 0
    if tot_true_rels > 0:
        recall_score = ret_rels / tot_true_rels
    return recall_score
