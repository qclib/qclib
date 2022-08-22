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
from qiskit import QuantumCircuit

from qclib.state_preparation import UCGInitialize
from qclib.util import get_state


class TestUCGInitialize(TestCase):
    """Test UCGInitialize"""

    def test_three_qubit_state_real(self):
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
        print(output_state @ state.T)
        self.assertTrue(np.allclose(output_state, state))

    def test_three_qubit_state_complex(self):
        """Test UCGInitialize with three qubits and index 7"""
        state = np.random.rand(8) + np.random.rand(8) * 1j
        state = state / np.linalg.norm(state)

        initialize = UCGInitialize.initialize
        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.x(1)
        circuit.x(2)

        initialize(circuit, state.tolist(), opt_params={"target_state": 7})

        output_state = get_state(circuit)
        print(output_state @ state.T)
        self.assertTrue(np.allclose(output_state, state))
