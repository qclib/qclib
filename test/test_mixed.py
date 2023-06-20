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
Tests for the mixed.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit.quantum_info import Statevector, DensityMatrix
from qclib.state_preparation import MixedInitialize

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestMixed(TestCase):

    def test_double_state(self):
        n_qubits = 5

        state_vector1 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        state_vector2 = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)

        ensemble = [state_vector1, state_vector2]
        circuit = MixedInitialize(ensemble, reset=False).definition

        zero = [1, 0]
        one = [0, 1]

        state_vector = 1/np.sqrt(2) * (np.kron(state_vector1, zero) + np.kron(state_vector2, one))
        state_vector = Statevector(state_vector)

        rho_classical = DensityMatrix(state_vector)
        rho_quantum = DensityMatrix(circuit)

        self.assertTrue(np.allclose(rho_quantum, rho_classical))

    def test_n_state(self):
        n_qubits = 3
        n_states = 5

        ensemble = []
        for _ in range(n_states):
            state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
            state_vector = state_vector / np.linalg.norm(state_vector)
            ensemble.append(state_vector)

        probabilities = np.random.rand(n_states)
        probabilities = probabilities / sum(probabilities)

        circuit_classical = MixedInitialize(
            ensemble,
            probabilities=probabilities,
            classical=True
        ).definition
        circuit_quantum = MixedInitialize(
            ensemble,
            probabilities=probabilities,
            classical=False
        ).definition

        rho_classical = DensityMatrix(circuit_classical)
        rho_quantum = DensityMatrix(circuit_quantum)

        self.assertTrue(np.allclose(rho_quantum, rho_classical))
