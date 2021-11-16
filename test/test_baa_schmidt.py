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
Tests for the baa_schmidt.py module.
"""

from unittest import TestCase
from typing import List
import numpy as np
from qiskit import ClassicalRegister, execute, Aer
from qclib.util import get_state
from qclib.state_preparation.baa_schmidt import initialize
from qclib.state_preparation.util.baa import adaptive_approximation

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestBaaSchmidt(TestCase):
    @staticmethod
    def calculate_state(vectors: List[np.ndarray]):
        state = np.ones(1)
        for vec in vectors:
            state = np.kron(vec, state)
        return state

    @staticmethod
    def get_counts(circuit):
        n_qubits = circuit.num_qubits
        classical_reg = ClassicalRegister(n_qubits)
        circuit.add_register(classical_reg)
        circuit.measure(list(range(n_qubits)), classical_reg)

        backend = Aer.get_backend('qasm_simulator')
        counts = execute(circuit, backend, shots=8192).result().get_counts()

        counts_with_zeros = {}
        for i in range(2**n_qubits):
            pattern = '{:0{}b}'.format(i, n_qubits)
            if pattern in counts:
                counts_with_zeros[pattern] = counts[pattern]
            else:
                counts_with_zeros[pattern] = 0.0

        sum_values = sum(counts.values())
        return [ value/sum_values for (key, value) in counts_with_zeros.items() ]

    def _test_initialize_loss(self, fidelity_loss):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, max_fidelity_loss=fidelity_loss)

        state = get_state(circuit)

        expected_state = state
        node = adaptive_approximation(state_vector, fidelity_loss)
        if node is not None:
            expected_state = TestBaaSchmidt.calculate_state(node.vectors)

        self.assertTrue(np.allclose(expected_state, state))

    def test_initialize_loss(self):
        for loss in range(1, 10):
            self._test_initialize_loss(loss/10)

    def test_initialize_no_loss(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_measurement_no_loss(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = TestBaaSchmidt.get_counts(circuit)

        self.assertTrue(np.allclose( np.power(np.abs(state_vector),2), state,
                        rtol=1e-01, atol=0.005))