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
""" Stopping Probability related functions"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def P_Rank(rank):  # pylint: disable=invalid-name
    """
    1/k Stopping probabilty.
    Args:
        rank (int): rank
    Returns:
        float: stopping probability at the given rank
    """
    return 1 / rank


def P_RBP(rank, theta=1):  # pylint: disable=invalid-name
    """
    Rank-biased precision based stopping probabiltiy.
    Args:
        rank (int): rank
        theta (int, optional): Defaults to 1. patience parameter theta.
    Returns:
        float: stopping probability at the given rank
    """

    return np.power(theta, rank - 1) * (1 - theta)


def P_DCG(rank, b=2):  # pylint: disable=invalid-name
    """
    Discounted Cumulative gain based stopping probability
    Args:
        rank (int): rank
        b (int, optional): Defaults to 1. log base.
    Returns:
        float: stopping probability at the given rank
    """

    def __log_n(x, n):  # pylint: disable=missing-docstring
        return np.log(x) / np.log(n)

    if b is None:
        utility_score = 1 / np.log(rank + 1)
    else:
        utility_score = 1 / __log_n(b + rank - 1, b)
    return utility_score
