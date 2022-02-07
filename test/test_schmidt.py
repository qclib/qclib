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
from itertools import combinations
from qiskit import ClassicalRegister, execute, transpile
from qiskit.providers.aer.backends import AerSimulator
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestSchmidt(TestCase):
    @staticmethod
    def mae(state, ideal):
        """
        Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)

    @staticmethod
    def fidelity(state1, state2):
        bra = np.conj(state1)
        ket = state2

        return np.abs(bra.dot(ket))**2

    @staticmethod
    def get_counts(circuit):
        n_qubits = circuit.num_qubits
        classical_reg = ClassicalRegister(n_qubits)
        circuit.add_register(classical_reg)
        circuit.measure(list(range(n_qubits)), classical_reg)

        backend = AerSimulator()
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

    def _test_initialize_mae(self, rank=0, max_mae=10**-15):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        qubits = list(range(5))
        partitions = combinations(qubits, 3)
        for partition in partitions:
            circuit = initialize(state_vector, low_rank=rank, partition=partition)

            state = get_state(circuit)

            self.assertTrue(TestSchmidt.mae(state, state_vector) < max_mae)

    def test_initialize_full_rank(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_initialize_ame(self):
        """ Test initialization of a absolutely maximally entangled state"""
        state_vector = [1, 1, 1, 1,1,-1,-1, 1, 1,-1,-1, 1, 1, 1,1,1,
                        1, 1,-1,-1,1,-1, 1,-1,-1, 1,-1, 1,-1,-1,1,1]
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_measurement_full_rank(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = TestSchmidt.get_counts(circuit)

        self.assertTrue(np.allclose( np.power(np.abs(state_vector),2), state,
                        rtol=1e-01, atol=0.005))

    def test_initialize_rank_5(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector, low_rank=5)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state)) # same as unitary.

    def test_initialize_rank_4(self):
        self._test_initialize_mae(4, max_mae=10**-14)

    def test_initialize_rank_3(self):
        self._test_initialize_mae(3, max_mae=0.04)

    def test_initialize_rank_2(self):
        self._test_initialize_mae(2, max_mae=0.06)

    def test_initialize_rank_1(self):
        self._test_initialize_mae(1, max_mae=0.09)

    def test_cnot_count_rank_1(self):
        # Builds a rank 1 state.
        state_vector = [1]
        for _ in range(5):
            vec = np.random.rand(2) + np.random.rand(2) * 1j
            vec = vec / np.linalg.norm(vec)
            state_vector = np.kron(state_vector, vec)

        circuit = initialize(state_vector)
        transpiled_circuit = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=3)

        # The cnots count must be zero (i.e. not present in the dictionary).
        self.assertTrue('cx' not in transpiled_circuit.count_ops())
