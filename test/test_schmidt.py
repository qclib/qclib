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

""" Test state preparation """

from unittest import TestCase
import numpy as np
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_state


class TestInitialize(TestCase):
    """ Testing qclib.state_preparation.schmidt """
    @staticmethod
    def mae(state, ideal):
        """
         Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)

    def test_initialize(self):
        """ Test state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(vector, state))

    def test_initialize_rank_5(self):
        """ Test rank 5 state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector, rank=5)

        state = get_state(circuit)

        self.assertTrue(np.allclose(vector, state)) # same as unitary.

    def test_initialize_rank_4(self):
        """ Test rank 4 state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector, rank=4)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state, vector) < 10**-15)

    def test_initialize_rank_3(self):
        """ Test rank 3 state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector, rank=3)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state,vector) < 0.04)

    def test_initialize_rank_2(self):
        """ Test rank 2 state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector, rank=2)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state,vector) < 0.055)

    def test_initialize_rank_1(self):
        """ Test rank 1 state vector initialization """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector, rank=1)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state,vector) < 0.08)
