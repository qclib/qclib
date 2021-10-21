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
Tests for the isometry.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import unitary_group
from qclib.isometry import decompose
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestIsometry(TestCase):

    # scheme='knill' isometry tests

    def test_state_preparation_real_knill(self):
        self._test_state_preparation_real('knill')

    def test_state_preparation_complex_knill(self):
        self._test_state_preparation_complex('knill')

    def test_isometry_knill(self):
        self._test_isometry('knill')

    def test_fixed_isometry_knill(self):
        self._test_fixed_isometry('knill')

    # scheme='ccd' (column-by-column decomposition) isometry tests

    def test_state_preparation_real_ccd(self):
        self._test_state_preparation_real('ccd')

    def test_state_preparation_complex_ccd(self):
        self._test_state_preparation_complex('ccd')

    def test_isometry_ccd(self):
        self._test_isometry('ccd')

    def test_fixed_isometry_ccd(self):
        self._test_fixed_isometry('ccd')

    # general functions for testing the schemes (knill and ccd).

    def _test_state_preparation_real(self, scheme):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = decompose(state_vector, scheme=scheme)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def _test_state_preparation_complex(self, scheme):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = decompose(state_vector, scheme=scheme)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def _test_isometry(self, scheme):
        log_lines = 4
        log_cols = 2
        isometry = unitary_group.rvs(2**log_lines)[:,:2**log_cols]

        gate = decompose(isometry, scheme=scheme)

        for j in range(2**log_cols):
            circuit = QuantumCircuit(log_lines)

            for i, bit in enumerate('{:0{}b}'.format(j, 1)[::-1]):
                if bit == '1':
                    circuit.x(i)

            circuit.append(gate, circuit.qubits)
            state = get_state(circuit)

            self.assertTrue(np.allclose(isometry[:, j], state))

    def _test_fixed_isometry(self, scheme):
        isometry = np.copy([[-0.5391073,  -0.12662419, -0.73739705, -0.38674956],
                            [ 0.15705405,  0.20566939,  0.32663193, -0.9090356 ],
                            [-0.77065035, -0.23739918,  0.59084039,  0.02544217],
                            [-0.30132273,  0.9409081,  -0.02155946,  0.15307435]])

        gate = decompose(isometry, scheme=scheme)

        state = get_state(gate)

        self.assertTrue(np.allclose(isometry[:, 0], state))

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(isometry[:, 1], state))

        circuit = QuantumCircuit(2)
        circuit.x(1)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(isometry[:, 2], state))

        circuit = QuantumCircuit(2)
        circuit.x([0,1])
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(isometry[:, 3], state))
