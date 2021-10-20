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
Tests for the schmidt.py module.
"""

from unittest import TestCase
import numpy as np
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestInitialize(TestCase):

    @staticmethod
    def mae(state, ideal):
        """
        Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)

    def _test_initialize_rank(self, rank, max_mae=10**-17):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, low_rank=rank)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state,state_vector) < max_mae)

    def test_initialize_full_rank(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_initialize_rank_5(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, low_rank=5)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state)) # same as unitary.

    def test_initialize_rank_4(self):
        self._test_initialize_rank(4, max_mae=10**-14)

    def test_initialize_rank_3(self):
        self._test_initialize_rank(3, max_mae=0.04)

    def test_initialize_rank_2(self):
        self._test_initialize_rank(2, max_mae=0.06)

    def test_initialize_rank_1(self):
        self._test_initialize_rank(1, max_mae=0.0825)
