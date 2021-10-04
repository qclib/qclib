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

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from scipy.stats import unitary_group
from qclib.isometry import decompose
from qclib.util import get_state

class TestInitialize(TestCase):

    def test_state_preparation_knill(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = decompose(a, scheme='knill')

        state = get_state(circuit)

        self.assertTrue(np.allclose(a, state))

    def test_unitary_knill(self):
        a = unitary_group.rvs(32)
        
        circuit = decompose(a, scheme='knill')

        state = get_state(circuit)

        self.assertTrue(np.allclose(a[:, 0], state))

    def test_isometry_knill(self):
        a = unitary_group.rvs(32)[:,:4]

        gate = decompose(a, scheme='knill')

        state = get_state(gate)

        self.assertTrue(np.allclose(a[:, 0], state))

        circuit = QuantumCircuit(5)
        circuit.x(0)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 1], state))

        circuit = QuantumCircuit(5)
        circuit.x(1)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 2], state))

        circuit = QuantumCircuit(5)
        circuit.x([0,1])
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 3], state))
    
    def test_compare_mottonen_isometry_knill(self):
        from qclib.state_preparation.mottonen import initialize

        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)
        
        circuit1 = decompose(a, scheme='knill')
        circuit2 = initialize(a)

        state1 = get_state(circuit1)
        state2 = get_state(circuit2)
        
        self.assertTrue(np.allclose(state1, state2))
        
    def test_compare_unitary_qsd_isometry_knill(self):
        from qclib.unitary import unitary

        a = unitary_group.rvs(32)
        
        circuit1 = decompose(a, scheme='knill')
        circuit2 = unitary(a, decomposition='qsd')

        state1 = get_state(circuit1)
        state2 = get_state(circuit2)
        
        self.assertTrue(np.allclose(state1, state2))
    
    def test_compare_unitary_csd_isometry_knill(self):
        from qclib.unitary import unitary

        a = unitary_group.rvs(32)
        
        circuit1 = decompose(a, scheme='knill')
        circuit2 = unitary(a, decomposition='csd')

        state1 = get_state(circuit1)
        state2 = get_state(circuit2)
        
        self.assertTrue(np.allclose(state1, state2))
    
    def test_compare_unitary_qiskit_isometry_knill(self):
        a = unitary_group.rvs(32)
        
        circuit1 = decompose(a, scheme='knill')
        circuit2 = QuantumCircuit(5)
        circuit2.unitary(a, list(range(5)))

        state1 = get_state(circuit1)
        state2 = get_state(circuit2)
                
        self.assertTrue(np.allclose(state1, state2))

    def test_schmidt_isometry_knill(self):
        a = np.copy([[-0.5391073,  -0.12662419, -0.73739705, -0.38674956],
                    [ 0.15705405,  0.20566939,  0.32663193, -0.9090356 ],
                    [-0.77065035, -0.23739918,  0.59084039,  0.02544217],
                    [-0.30132273,  0.9409081,  -0.02155946,  0.15307435]])
        
        gate = decompose(a, scheme='knill')
        
        state = get_state(gate)
        
        self.assertTrue(np.allclose(a[:, 0], state))

        circuit = QuantumCircuit(2)
        circuit.x(0)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 1], state))

        circuit = QuantumCircuit(2)
        circuit.x(1)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 2], state))

        circuit = QuantumCircuit(2)
        circuit.x([0,1])
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(a[:, 3], state))