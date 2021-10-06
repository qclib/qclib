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

from unittest import TestCase
import numpy as np
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_state
from qiskit import transpile


class TestInitialize(TestCase):

    @staticmethod
    def mae(state, ideal):
        """
         Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)
        
    def test_initialize(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a)

        state = get_state(circuit)
        
        self.assertTrue(np.allclose(a, state))
    
    def test_initialize_rank_5(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, low_rank=5)

        state = get_state(circuit)
        
        self.assertTrue(np.allclose(a, state)) # same as unitary.

    def test_initialize_rank_4(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, low_rank=4)

        state = get_state(circuit)
        
        self.assertTrue(TestInitialize.mae(state,a) < 10**-14)

    def test_initialize_rank_3(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, low_rank=3)

        state = get_state(circuit)

        self.assertTrue(TestInitialize.mae(state,a) < 0.04)

    def test_initialize_rank_2(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, low_rank=2)

        state = get_state(circuit)
        
        self.assertTrue(TestInitialize.mae(state,a) < 0.055)
    
    def test_initialize_rank_1(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, low_rank=1)

        state = get_state(circuit)
        
        self.assertTrue(TestInitialize.mae(state,a) < 0.0825)