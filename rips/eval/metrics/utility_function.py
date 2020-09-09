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
""" Utility Functions used in evaluation metric"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np


def utility_DCG(grade):  # pylint: disable=invalid-name
    """
    Discounted Cumulative gain based utility function.
    Args:
      grade (float): grade value.
    Returns:
      float: utility score for the grade
    """
    return np.power(2, grade) - 1


def utility_Prec(grade):  # pylint: disable=invalid-name
    """
    Simple precision based or identity utility function.
    Args:
      grade (float): grade value.
    Returns:
      float: utility score for the grade
    """
    return grade


def utility_Binary(grade):  # pylint: disable=invalid-name
    """
    Binary utility function, that converts any grades into binary utility.
    Args:
      grade (float): grade value.
    Returns:
      float: utility score for the grade
    """
    return 1 if grade > 0 else 0
