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
from qclib.unitary import unitary, _compute_gates, cnot_count, _build_qr_gate_sequence
from qclib.util import get_state

class TestUnitary(TestCase):
    """ Testing qclib.unitary """
    def _test_unitary(self, decomposition, n_qubits, iso=0, apply_a2=False):
        """ Testing qclib.unitary gate """
        unitary_matrix = unitary_group.rvs(2**n_qubits)
        gate = unitary(unitary_matrix, decomposition, iso, apply_a2)
        gate = qiskit.transpile(gate, basis_gates=['u', 'cx'])

        for i in range(2**n_qubits):
            circuit = qiskit.QuantumCircuit(n_qubits)

            for j, bit in enumerate(f'{i:0{n_qubits}b}'[::-1]):
                if bit == '1':
                    circuit.x(j)

            circuit.append(gate.to_instruction(), circuit.qubits)
            state = get_state(circuit)
            self.assertTrue(np.allclose(unitary_matrix[:, i], state))

    def _test_counting(self, decomposition, n_qubits, iso=0, apply_a2=False):
        unitary_matrix = unitary_group.rvs(2**n_qubits)

        n_cx_exact = cnot_count(unitary_matrix, decomposition, 'exact', iso, apply_a2)
        n_cx_estimate = cnot_count(unitary_matrix, decomposition, 'estimate', iso, apply_a2)

        self.assertTrue(n_cx_exact == n_cx_estimate)

    def test_compute_gates(self):
        """ test auxiliar function compute gates"""
        gate1 = unitary_group.rvs(8)
        gate2 = unitary_group.rvs(8)

        diag, v_gate, w_gate = _compute_gates(gate1, gate2)
        calc1 = v_gate @ np.diag(diag) @ w_gate
        calc2 = v_gate @ np.diag(diag).conj().T @ w_gate
        self.assertTrue(np.allclose(calc1, gate1))
        self.assertTrue(np.allclose(calc2, gate2))

    # CSD

    def test_unitary_csd(self):
        """ Testing qclib.unitary csd"""
        for n_qubits in range(2, 5):
            self._test_unitary('csd', n_qubits)

    def test_counting_csd(self):
        """ Testing qclib.unitary.cnot_count csd"""
        for n_qubits in range(2, 6):
            self._test_counting('csd', n_qubits)

    # QR
    def test_unitary_qr(self):
        """ Testing qclib.unitary csd"""
        for n_qubits in range(2, 5):
            self._test_unitary('qr', n_qubits)

    def test_qr_gate_sequence(self):
        n_qubits = 2
        unitary_matrix = unitary_group.rvs(2**n_qubits)
        gate_sequence = _build_qr_gate_sequence(unitary_matrix, n_qubits)
        
        unitary_rebuilt = np.eye(2**n_qubits)
        for g in gate_sequence:
            unitary_rebuilt = g @ unitary_rebuilt

        self.assertTrue(np.allclose(unitary_rebuilt, unitary_matrix))

    # QSD

    def test_unitary_qsd(self):
        """ Testing qclib.unitary qsd"""
        for n_qubits in range(2, 5):
            self._test_unitary('qsd', n_qubits, 0, False)

        for n_qubits in range(2, 5):
            self._test_unitary('qsd', n_qubits, 0, True)

    def test_counting_qsd(self):
        """ Testing qclib.unitary.cnot_count qsd"""
        for n_qubits in range(2, 6):
            self._test_counting('qsd', n_qubits, 0, False)

        for n_qubits in range(2, 6):
            self._test_counting('qsd', n_qubits, 0, True)

    def test_unitary_qsd_count(self):
        """ Testing qclib.unitary 4 qubits gate qsd"""
        unitary_matrix = unitary_group.rvs(16)
        gate = unitary(unitary_matrix, 'qsd')
        gate = qiskit.transpile(gate, basis_gates=['u', 'cx'])
        state = get_state(gate)
        n_cx = gate.count_ops()['cx']
        self.assertTrue(n_cx <= 100)
        self.assertTrue(np.allclose(unitary_matrix[:, 0], state))
