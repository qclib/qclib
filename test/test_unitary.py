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

    def test_compute_gates_fixed(self):
        """ test auxiliar funciont compute gates"""
        # Matrices gate1 and gate2 are not unitary.
        gate1 = np.array(
                [[ 1.39499204e-01-1.02062065e-01j, -3.73248642e-01-6.00570938e-01j,
                   5.83554297e-01-1.84683236e-01j, -2.69576508e-01+1.51024165e-01j],
                 [-4.71845544e-01-6.91006095e-01j, -7.85046229e-17+0.00000000e+00j,
                   2.78615481e-01+1.87070569e-01j,  4.00199035e-01-1.64593766e-01j],
                 [ 7.04556392e-02-4.84927805e-01j,  0.00000000e+00+9.81307787e-18j,
                  -3.29381826e-01-1.72042433e-01j, -6.36139910e-01-4.65957139e-01j],
                 [-1.02062065e-01-1.39499204e-01j,  6.00570938e-01-3.73248642e-01j,
                  -1.84683236e-01-5.83554297e-01j,  1.51024165e-01+2.69576508e-01j]])
        gate2 = np.array(
                [[ 1.72752243e-01-5.77541244e-03j,  2.97965199e-02-7.06478710e-01j,
                   5.86222022e-01+1.76031973e-01j, -3.07814277e-01-2.70215300e-02j],
                 [-8.22782535e-01+1.52172736e-01j,  4.11897902e-17+7.20821328e-17j,
                   2.83747390e-01-1.79190967e-01j,  1.12670242e-02-4.32577657e-01j],
                 [-4.15037537e-01-2.60504925e-01j, -1.02974475e-17+4.11897902e-17j,
                  -2.90528108e-01+2.31698952e-01j, -6.83375688e-01+3.93430693e-01j],
                 [-5.77541244e-03-1.72752243e-01j,  7.06478710e-01+2.97965199e-02j,
                   1.76031973e-01-5.86222022e-01j, -2.70215300e-02+3.07814277e-01j]])

        # Let's find the closest unitary matrices to gate1 and gate2.
        def closest_unitary(matrix):
            svd_u,_,svd_v = np.linalg.svd(matrix)
            return svd_u.dot(svd_v)
        gate1 = closest_unitary(gate1)
        gate2 = closest_unitary(gate2)

        diag, v_gate, w_gate = _compute_gates(gate1, gate2)
        calc1 = v_gate @ np.diag(diag) @ w_gate
        calc2 = v_gate @ np.diag(diag).conj().T @ w_gate
        self.assertTrue(np.allclose(calc1, gate1))
        self.assertTrue(np.allclose(calc2, gate2))
