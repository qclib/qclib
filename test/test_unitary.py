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

""" Test creation of quantum circuits from matrices """

from unittest import TestCase
from scipy.stats import unitary_group
import numpy as np
import qiskit
from qclib.unitary import unitary, _compute_gates
from qclib.util import get_state

class TestUnitary(TestCase):
    """ Testing qclib.unitary """
    def test_unitary_csd_2qubits(self):
        """ Testing qclib.unitary with 2 qubits gate"""
        unitary_matrix = unitary_group.rvs(4)
        gate = unitary(unitary_matrix)
        state = get_state(gate)

        self.assertTrue(np.allclose(unitary_matrix[:, 0], state))

    def test_unitary_csd_5qubits(self):
        """ Testing qclib.unitary 5 qubits gate"""
        unitary_matrix = unitary_group.rvs(32)
        gate = unitary(unitary_matrix)
        state = get_state(gate)
        self.assertTrue(np.allclose(unitary_matrix[:, 0], state))

    def test_unitary_qsd_4qubits(self):
        """ Testing qclib.unitary 4 qubits gate qsd"""
        unitary_matrix = unitary_group.rvs(16)
        gate = unitary(unitary_matrix, 'qsd')
        state = get_state(gate)
        self.assertTrue(np.allclose(unitary_matrix[:, 0], state))

        circuit = qiskit.QuantumCircuit(4)
        circuit.x(0)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(unitary_matrix[:, 1], state))

        circuit = qiskit.QuantumCircuit(4)
        circuit.x(1)
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(unitary_matrix[:, 2], state))

        circuit = qiskit.QuantumCircuit(4)

        circuit.x([0, 1, 2, 3])
        circuit.append(gate, circuit.qubits)
        state = get_state(circuit)
        self.assertTrue(np.allclose(unitary_matrix[:, 15], state))

    def test_unitary_qsd_5qubits(self):
        """ Testing qclib.unitary 5 qubits gate qsd"""
        unitary_matrix = unitary_group.rvs(16)
        gate = unitary(unitary_matrix, 'qsd')
        state = get_state(gate)
        transpiled_circuit = qiskit.transpile(gate, basis_gates=['u', 'cx'])
        n_cx = transpiled_circuit.count_ops()['cx']
        self.assertTrue(n_cx <= 120)
        self.assertTrue(np.allclose(unitary_matrix[:, 0], state))

    def test_compute_gates(self):
        """ test auxiliar funciont compute gates"""
        gate1 = unitary_group.rvs(8)
        gate2 = unitary_group.rvs(8)

        diag, v_gate, w_gate = _compute_gates(gate1, gate2)
        calc1 = v_gate @ np.diag(diag) @ w_gate
        calc2 = v_gate @ np.diag(diag).conj().T @ w_gate
        self.assertTrue(np.allclose(calc1, gate1))
        self.assertTrue(np.allclose(calc2, gate2))
