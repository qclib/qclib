# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
	Default qclib state initialization functions.
	For dense state vectors, use ``initialize``.
	For sparse states, use ``sparse_initialize``.
"""

from qclib.state_preparation.schmidt import initialize as dense_init
from qclib.state_preparation.pivot import PivotStatePreparation
sparse_init = PivotStatePreparation.initialize

# pylint: disable=maybe-no-member


def initialize(state):
    """ Initialize dense state """
    return dense_init(state)


def sparse_initialize(state):
    """ Initialize sparse state """
    return sparse_init(state)
