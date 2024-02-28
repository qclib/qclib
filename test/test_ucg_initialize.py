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
    Tests for the bergholm state preparation
"""
from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit, transpile
from qclib.state_preparation import UCGInitialize
from qclib.state_preparation.ucge import UCGEInitialize
from qclib.util import get_state


class TestUCGInitialize(TestCase):
    """Test UCGInitialize"""

    def _test_ucg(self, n_qubits):
        state = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state = state / np.linalg.norm(state)

        for target_state in range(2**n_qubits):
            gate = UCGInitialize(state.tolist(),
                                    opt_params={
                                        "target_state": target_state
                                    }
                                ).definition

            circuit = QuantumCircuit(n_qubits)

            for j, bit in enumerate(f'{target_state:0{n_qubits}b}'[::-1]):
                if bit == '1':
                    circuit.x(j)

            circuit.append(gate, circuit.qubits)
            output_state = get_state(circuit)

            self.assertTrue(np.allclose(state, output_state))

    def _test_ucg_preserve(self, n_qubits):
        state = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j

        for target_state in range(1, 2**n_qubits):
            state[target_state - 1] = 0
            state = state / np.linalg.norm(state)

            gate = UCGInitialize(state.tolist(),
                                    opt_params={
                                        "target_state": target_state,
                                        "preserve_previous": True
                                    }
                                ).definition

            circuit = QuantumCircuit(n_qubits)

            for j, bit in enumerate(f'{target_state:0{n_qubits}b}'[::-1]):
                if bit == '1':
                    circuit.x(j)

            circuit.append(gate, circuit.qubits)
            output_state = get_state(circuit)

            self.assertTrue(np.allclose(output_state, state))

    def test_ucg(self):
        """Test UCGInitialize"""
        for n_qubits in range(3, 5):
            self._test_ucg(n_qubits)

    def test_ucg_preserve(self):
        """Test UCGInitialize with `preserve_previous`"""
        for n_qubits in range(3, 5):
            self._test_ucg_preserve(n_qubits)

    def test_real(self):
        """Test UCGInitialize with four qubits and index 10"""
        nqubits = 4
        state = np.random.rand(2 ** nqubits)
        state[0] = 0
        state[1] = 0
        state = state / np.linalg.norm(state)

        initialize = UCGInitialize.initialize
        circuit = QuantumCircuit(nqubits)
        circuit.x(1)
        circuit.x(3)

        initialize(circuit, state.tolist(), opt_params={"target_state": 10})

        output_state = get_state(circuit)
        self.assertTrue(np.allclose(output_state, state))

    def test_ucg_disentangled(self):
        num_qubits = 6
        qc = QuantumCircuit(num_qubits)

        input_state1 = np.random.rand(2 ** 3)
        input_state1 = input_state1 / np.linalg.norm(input_state1)
        input_state2 = np.random.rand(2 ** 3)
        input_state2 = input_state2 / np.linalg.norm(input_state2)

        UCGInitialize.initialize(qc, input_state1, [0, 2, 3])
        UCGInitialize.initialize(qc, input_state2, [1, 4, 5])

        params = get_state(qc)

        circuit = QuantumCircuit(6)
        UCGEInitialize.initialize(circuit, params.tolist())
        params2 = get_state(circuit)

        circuit_tranpiled = transpile(circuit, basis_gates=['u', 'cx'])
        qc_transpiled = transpile(qc, basis_gates=['u', 'cx'])

        self.assertTrue(np.allclose(params, params2))
        self.assertTrue(circuit_tranpiled.depth() <= qc_transpiled.depth())

    def test_ucg_disentangled_complex(self):
        num_qubits = 8
        qc = QuantumCircuit(num_qubits)

        real_part1 = np.random.rand(2 ** 4)
        imag_part1 = np.random.rand(2 ** 4)
        input_state1 = real_part1 + 1j * imag_part1

        input_state1 = input_state1 / np.linalg.norm(input_state1)

        real_part2 = np.random.rand(2 ** 4)
        imag_part2 = np.random.rand(2 ** 4)
        input_state2 = real_part2 + 1j * imag_part2

        input_state2 = input_state2 / np.linalg.norm(input_state2)

        UCGInitialize.initialize(qc, input_state1, [0, 2, 3, 7])
        UCGInitialize.initialize(qc, input_state2, [1, 4, 5, 6])

        params = get_state(qc)

        circuit = QuantumCircuit(8)
        UCGInitialize.initialize(circuit, params.tolist())
        params2 = get_state(circuit)

        self.assertTrue(np.allclose(params, params2))