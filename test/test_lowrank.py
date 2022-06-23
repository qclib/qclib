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
Tests for the lowrank.py module.
"""

from unittest import TestCase
from itertools import combinations
import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, execute, transpile
from qiskit.providers.aer.backends import AerSimulator
from qclib.state_preparation import LowRankInitialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestLowRank(TestCase):
    @staticmethod
    def mae(state, ideal):
        """
        Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)

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
            pattern = f'{i:0{n_qubits}b}'
            if pattern in counts:
                counts_with_zeros[pattern] = counts[pattern]
            else:
                counts_with_zeros[pattern] = 0.0

        sum_values = sum(counts.values())
        return [value/sum_values for (key, value) in counts_with_zeros.items()]

    def _test_initialize_mae(self, rank=0, max_mae=10**-15):
        n_qubits = 5
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        qubits = list(range(n_qubits))
        partitions = combinations(qubits, (n_qubits+1)//2)
        for partition in partitions:
            circuit = QuantumCircuit(n_qubits)
            lr_params = {'lr': rank, 'partition': partition}
            LowRankInitialize.initialize(circuit, state_vector, lr_params=lr_params)

            state = get_state(circuit)

            self.assertTrue(TestLowRank.mae(state, state_vector) < max_mae)

    def _test_initialize_full_rank(self, n_qubits):
        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        qubits = list(range(n_qubits))
        partitions = combinations(qubits, (n_qubits+1)//2)
        for partition in partitions:
            circuit = QuantumCircuit(n_qubits)
            lr_params = {'partition': partition}
            LowRankInitialize.initialize(circuit, state_vector, lr_params=lr_params)

            state = get_state(circuit)

            self.assertTrue(np.allclose(state_vector, state))

    def test_initialize_full_rank_7(self):
        self._test_initialize_full_rank(7)

    def test_initialize_full_rank_6(self):
        self._test_initialize_full_rank(6)

    def test_initialize_rank_5(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = QuantumCircuit(5)
        LowRankInitialize.initialize(circuit, state_vector, lr_params={'lr': 5})

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

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

        circuit = QuantumCircuit(5)
        LowRankInitialize.initialize(circuit, state_vector)
        transpiled_circuit = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=3)

        self.assertTrue('cx' not in transpiled_circuit.count_ops())
