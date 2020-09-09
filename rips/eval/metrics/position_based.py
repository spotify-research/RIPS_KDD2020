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
""" Position Based Ranking Metrics
        nDCG
        MRR
        Precision
        RBP
"""
from typing import List

import numpy as np


def mrr(judged_ranked_list: List[float]) -> List[float]:
    """
    Computes the Mean Reciprocal Rank (MRR).
    Args:
        judged_ranked_list (List[int]): judged ranked list (list of 1s & 0s).
    Returns:
        List[float]: mrr at requested rank cut offs
    """

    judged_ranked_list = np.asarray(judged_ranked_list)
    score = 0
    if np.sum(judged_ranked_list) > 0:
        score = 1 / ((judged_ranked_list > 0).argmax(axis=0) + 1)
    return score


def precision(judged_ranked_list: List[float], cut_off: List[int]) -> List[float]:
    """
    computes the precision of a given judged ranked list
    Args:
        judged_ranked_list (List[int]): judged ranked list (list of 1s & 0s).
        cut_off (List[int]): list of rank cut offs at which
                             precision must be reported.
    Returns:
        List[float]: precision at requested rank cut offs.
    """
    if max(cut_off) - len(judged_ranked_list) > 0:
        judged_ranked_list = np.pad(
            judged_ranked_list,
            pad_width=[0, max(cut_off) - len(judged_ranked_list)],
            mode="constant",
            constant_values=0,
        )
    else:
        judged_ranked_list = np.asarray(judged_ranked_list[0 : max(cut_off)])
    index = np.asarray(cut_off) - 1
    prec = np.cumsum(judged_ranked_list) / np.asarray(range(1, max(cut_off) + 1))
    return prec[index]


def dcg(judged_ranked_list: List[float], cut_off: List[int], b: int = None) -> np.ndarray:
    """Computes the Discounted Cumulative Gain
    :param judged_ranked_list: judged ranked list (list of 1s & 0s).
    :param cut_off: list of rank cut offs at which precision must be reported.
    :param b: log base value.
    :return: dcg at requested rank cut offs
    """
    from .stopping_probability import P_DCG
    from .utility_function import utility_DCG

    if max(cut_off) - len(judged_ranked_list) > 0:
        judged_ranked_list = np.pad(
            judged_ranked_list,
            pad_width=[0, max(cut_off) - len(judged_ranked_list)],
            mode="constant",
            constant_values=0,
        )
    else:
        judged_ranked_list = np.asarray(judged_ranked_list[0 : max(cut_off)])

    dcg_score = [(P_DCG(rank + 1, b) * utility_DCG(grade)) for rank, grade in enumerate(judged_ranked_list)]
    index = np.asarray(cut_off) - 1
    return np.cumsum(dcg_score)[index]


def ndcg(judged_ranked_list: List[float], ideal_grades: List[float], cut_off: List[int], b: int = None) -> List[float]:
    """Computes the normalized Discounted Cumulative Gain
    :param judged_ranked_list: judged ranked list (list of 1s & 0s).
    :param ideal_grades: ideal ranked list (list of 1s & 0s).
    :param cut_off: list of rank cut offs at which precision must be reported.
    :param b: log base value.
    :return: ndcg at requested rank cut offs
    """
    ideal_dcg = dcg(ideal_grades, cut_off, b)
    if np.sum(ideal_dcg) == 0:
        ndcg_score = np.zeros(len(cut_off))
    else:
        ndcg_score = dcg(judged_ranked_list, cut_off, b) / dcg(ideal_grades, cut_off, b)
    return ndcg_score


def rbp(judged_ranked_list: List[float], cut_off: List[int], theta: float = 0.5) -> List[float]:
    """Computes the Rank Biased Precision
    :param judged_ranked_list: judged ranked list (list of 1s & 0s).
    :param cut_off: list of rank cut offs at which precision must be reported.
    :param theta: theta paramter in rbp.
    :return: rbp at requested rank cut offs
    """
    from .stopping_probability import P_RBP
    from .utility_function import utility_Binary

    if max(cut_off) - len(judged_ranked_list) > 0:
        judged_ranked_list = np.pad(
            judged_ranked_list,
            pad_width=[0, max(cut_off) - len(judged_ranked_list)],
            mode="constant",
            constant_values=0,
        )
    else:
        judged_ranked_list = np.asarray(judged_ranked_list[0 : max(cut_off)])

    rbp_score = [(P_RBP(rank + 1, theta) * utility_Binary(grade)) for rank, grade in enumerate(judged_ranked_list)]
    index = np.asarray(cut_off) - 1
    return (np.cumsum(rbp_score))[index]
