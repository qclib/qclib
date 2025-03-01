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
Tests for UCGEInitialize
https://arxiv.org/abs/2409.05618
"""
from unittest import TestCase
import random
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qclib.state_preparation import UCGInitialize
from qclib.state_preparation import UCGEInitialize
from qclib.util import logical_swap

class TestUCGEInitialize(TestCase):
    """Test UCGEInitialize"""

    def _test_compare_ucg_bipartition(self, num_qubits, input_vector1, input_vector2):
        qubit_order = list(range(num_qubits))
        random.shuffle(qubit_order)

        input_vector = np.kron(input_vector1, input_vector2)
        input_vector = logical_swap(input_vector, qubit_order)

        circuit1 = QuantumCircuit(num_qubits)
        UCGInitialize.initialize(circuit1, input_vector)
        state1 = Statevector(circuit1)

        circuit2 = QuantumCircuit(num_qubits)
        UCGEInitialize.initialize(circuit2, input_vector)
        state2 = Statevector(circuit2)

        circuit1_transpiled = transpile(circuit1, basis_gates=["u", "cx"])
        circuit2_transpiled = transpile(circuit2, basis_gates=["u", "cx"])

        self.assertTrue(np.allclose(input_vector, state2))
        self.assertTrue(np.allclose(state1, state2))
        self.assertTrue(
            2 * circuit1_transpiled.depth() ** (1 / 2) > circuit2_transpiled.depth()
        )

    def test_compare_ucg_bipartition_real(self):
        num_qubits1 = 3
        num_qubits2 = 4
        num_qubits = num_qubits1 + num_qubits2

        input_vector1 = np.random.rand(2**num_qubits1)
        input_vector1 = input_vector1 / np.linalg.norm(input_vector1)

        input_vector2 = np.random.rand(2**num_qubits2)
        input_vector2 = input_vector2 / np.linalg.norm(input_vector2)

        self._test_compare_ucg_bipartition(num_qubits, input_vector1, input_vector2)

    def test_compare_ucg_bipartition_complex(self):
        num_qubits1 = 3
        num_qubits2 = 4
        num_qubits = num_qubits1 + num_qubits2

        real_part = np.random.rand(2**num_qubits1)
        imag_part = np.random.rand(2**num_qubits1)
        input_vector1 = real_part + 1j * imag_part
        input_vector1 = input_vector1 / np.linalg.norm(input_vector1)

        real_part = np.random.rand(2**num_qubits2)
        imag_part = np.random.rand(2**num_qubits2)
        input_vector2 = real_part + 1j * imag_part
        input_vector2 = input_vector2 / np.linalg.norm(input_vector2)

        self._test_compare_ucg_bipartition(num_qubits, input_vector1, input_vector2)

    def test_3_partition_complex(self):
        num_qubits1 = 2
        num_qubits2 = 3
        num_qubits3 = 1
        num_qubits = num_qubits1 + num_qubits2 + num_qubits3

        real_part = np.random.rand(2**num_qubits1)
        imag_part = np.random.rand(2**num_qubits1)
        input_vector1 = real_part + 1j * imag_part
        input_vector1 = input_vector1 / np.linalg.norm(input_vector1)

        real_part = np.random.rand(2**num_qubits2)
        imag_part = np.random.rand(2**num_qubits2)
        input_vector2 = real_part + 1j * imag_part
        input_vector2 = input_vector2 / np.linalg.norm(input_vector2)

        real_part = np.random.rand(2**num_qubits3)
        imag_part = np.random.rand(2**num_qubits3)
        input_vector3 = real_part + 1j * imag_part
        input_vector3 = input_vector3 / np.linalg.norm(input_vector3)

        input_vector = np.kron(np.kron(input_vector1, input_vector2), input_vector3)

        circuit1 = QuantumCircuit(num_qubits)
        UCGInitialize.initialize(circuit1, input_vector)
        state1 = Statevector(circuit1)

        circuit2 = QuantumCircuit(num_qubits)
        UCGEInitialize.initialize(circuit2, input_vector)
        state2 = Statevector(circuit2)

        circuit1_transpiled = transpile(circuit1, basis_gates=["u", "cx"])
        circuit2_transpiled = transpile(circuit2, basis_gates=["u", "cx"])

        self.assertTrue(np.allclose(input_vector, state2))
        self.assertTrue(np.allclose(state1, state2))
        self.assertTrue(2 * circuit1_transpiled.depth() ** (1 / 3) > circuit2_transpiled.depth())

    def test_minimal_complex(self):
        np.random.seed(1)
        n_qubits = 2

        # Creates a product state
        state = [1]
        for _ in range(n_qubits):
            state_one_qubit = np.random.rand(2) + np.random.rand(2) * 1j
            state_one_qubit = state_one_qubit / np.linalg.norm(state_one_qubit)
            state = np.kron(state, state_one_qubit)

        ucge_circ = UCGEInitialize(state).definition
        transpiled_ucge_circ = transpile(ucge_circ, basis_gates=["u", "cx"])
        ucge_depth = transpiled_ucge_circ.depth()

        self.assertTrue(ucge_depth == 1)
