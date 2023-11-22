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
Tests for the topdown.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, execute
from qiskit_aer import AerSimulator
from qclib.state_preparation import TopDownInitialize
from qclib.util import get_state, measurement

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

backend = AerSimulator()
SHOTS = 8192


class TestTopDown(TestCase):

    @staticmethod
    def topdown_experiment(state):
        circuit = TopDownInitialize(state).definition

        n_qubits = int(np.log2(len(state)))
        classical_reg = ClassicalRegister(n_qubits)
        circuit.add_register(classical_reg)

        return measurement(circuit, n_qubits, classical_reg, backend, SHOTS)

    def test_topdown_state_real(self):
        state_vector = np.random.rand(32)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = TopDownInitialize(state_vector).definition

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_topdown_state_complex(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = TopDownInitialize(state_vector).definition

        state = get_state(circuit)

        self.assertTrue(np.allclose(np.imag(state_vector), np.imag(state)))
        self.assertTrue(np.allclose(np.real(state_vector), np.real(state)))

    def test_topdown_measure(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        state = TestTopDown.topdown_experiment(state_vector)

        self.assertTrue(np.allclose(np.power(np.abs(state_vector), 2), state,
                                    rtol=1e-01, atol=0.005))

    def test_topdown_fixed_state(self):
        state_vector = [
            0, np.sqrt(2 / 8) * np.exp(-1.0j * np.pi / 7),
               np.sqrt(3 / 8) * np.exp(-1.0j * np.pi / 3), 0,
            0, 0, np.sqrt(3 / 8), 0
        ]

        circuit = TopDownInitialize(state_vector).definition

        state = get_state(circuit)

        self.assertTrue(np.allclose(np.imag(state_vector), np.imag(state)))
        self.assertTrue(np.allclose(np.real(state_vector), np.real(state)))
