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
Tests for the svd.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from qclib.state_preparation.svd import SVDInitialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestSVDInitialize(TestCase):

    def _test_initialize(self, n_qubits):
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = QuantumCircuit(n_qubits)

        SVDInitialize.initialize(circuit, state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_initialize_6(self):
        self._test_initialize(6)

    def test_initialize_5(self):
        self._test_initialize(5)
