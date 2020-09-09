#! /usr/bin/env python

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
import os

from setuptools import find_packages
from setuptools import setup

HERE = os.path.abspath(os.path.dirname(__file__))

REQUIRED_PACKAGES = [
    "tensorflow==2.3.0",
    "matplotlib==3.3.1",
    "apache-beam[gcp]==2.23.0",
    "jupyter==1.0.0",
    "pytest==6.0.1",
    "fire==0.3.1",
    "scipy==1.4.1",
    "numpy==1.16.0",
    "pandas==1.1.1",
    "tqdm==4.48.2",
]

setup(
    name="rips",
    version="0.0.1",
    description="KDD 2020 RIPS",
    url="",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
    ],
    zip_safe=False,
    install_requires=REQUIRED_PACKAGES,
)
