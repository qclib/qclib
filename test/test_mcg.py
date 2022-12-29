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

""" Test qclib.gate.mcg """

from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qclib.util import get_cnot_count, get_depth
from qclib.gates.mcg import mcg
from qclib.gates.mcg import linear_depth_any_mcsu2


# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


QuantumCircuit.mcg = mcg


class TestMcg(TestCase):
    """ Testing qclib.gate.mcg """

    def _build_su2_circuit(self, alpha, beta, n_qubits):
        length = np.linalg.norm([alpha, beta])
        su2 = np.array([
            [alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]
        ]) / length

        return self._build_circuit(su2, n_qubits)

    def _build_circuit(self, unitary, n_qubits):
        mcg_circuit = QuantumCircuit(n_qubits)
        qiskit_circuit = QuantumCircuit(1)
        if n_qubits > 1:
            controls = list(range(n_qubits - 1))
            target = n_qubits - 1
            ctrl_state = '0' * (n_qubits-1)

            mcg_circuit.mcg(unitary, controls, target, ctrl_state=ctrl_state)

            qiskit_circuit.unitary(unitary, 0)
            qiskit_circuit = qiskit_circuit.control(
                num_ctrl_qubits=n_qubits - 1, ctrl_state=ctrl_state
            )
        else:
            mcg_circuit.mcg(unitary, [], 0)
            qiskit_circuit.unitary(unitary, 0)

        return mcg_circuit, qiskit_circuit

    def _su2_count(self, alpha, beta, n_qubits):
        mcg_circuit, qiskit_circuit = self._build_su2_circuit(alpha, beta, n_qubits)
        
        # Count cnots
        mcg_cx = get_cnot_count(mcg_circuit)
        qiskit_cx = get_cnot_count(qiskit_circuit)
        
        self.assertTrue(mcg_cx <= qiskit_cx)

    def _su2_depth(self, alpha, beta, n_qubits):
        mcg_circuit, qiskit_circuit = self._build_su2_circuit(alpha, beta, n_qubits)

        # Count cnots
        mcg_dp = get_depth(mcg_circuit)
        qiskit_dp = get_depth(qiskit_circuit)
        
        self.assertTrue(mcg_dp <= qiskit_dp)

    def _su2_compare(self, alpha, beta, n_qubits):
        mcg_circuit, qiskit_circuit = self._build_su2_circuit(alpha, beta, n_qubits)

        # Compare
        mcg_op = Operator(mcg_circuit).data
        qiskit_op = Operator(qiskit_circuit).data

        self.assertTrue(np.allclose(mcg_op, qiskit_op))

    def _su2_iten_count(self, alpha, beta, n_qubits):
        length = np.linalg.norm([alpha, beta])
        su2 = np.array([
            [alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]
        ]) / length

        su2_iten_circuit = QuantumCircuit(n_qubits)

        linear_depth_any_mcsu2(
            su2_iten_circuit,
            unitary=su2,
            controls=list(range(n_qubits - 1)),
            target=n_qubits - 1
        )

        su2_iten_cx = get_cnot_count(su2_iten_circuit)

        if n_qubits >= 8:
            constant = 0

            if n_qubits % 2 == 0:
                constant = 88
            else:
                constant = 92

            self.assertLessEqual(su2_iten_cx, 28*n_qubits - constant)

    def _su2_iten_compare(self, alpha, beta, n_qubits):
        length = np.linalg.norm([alpha, beta])
        su2 = np.array([
            [alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]
        ]) / length

        su2_iten_circuit = QuantumCircuit(n_qubits)
        su2_qiskit_circuit = QuantumCircuit(1)

        linear_depth_any_mcsu2(
            su2_iten_circuit,
            unitary=su2,
            controls=list(range(n_qubits - 1)),
            target=n_qubits - 1
        )

        su2_qiskit_circuit.unitary(su2, 0)

        if n_qubits > 1:
            su2_qiskit_circuit = su2_qiskit_circuit.control(num_ctrl_qubits=n_qubits - 1)

        # Compare
        iten_op = Operator(su2_iten_circuit).data
        qiskit_op = Operator(su2_qiskit_circuit).data

        self.assertTrue(np.allclose(iten_op, qiskit_op))

    def _u2_count(self, unitary, n_qubits):
        mcg_circuit, qiskit_circuit = self._build_circuit(unitary, n_qubits)

        # Count cnots
        mcg_cx = get_cnot_count(mcg_circuit)
        qiskit_cx = get_cnot_count(qiskit_circuit)
        
        self.assertTrue(mcg_cx <= qiskit_cx)

    def _u2_compare(self, unitary, n_qubits):
        mcg_circuit, qiskit_circuit = self._build_circuit(unitary, n_qubits)

        # Compare
        mcg_op = Operator(mcg_circuit).data
        qiskit_op = Operator(qiskit_circuit).data

        self.assertTrue(np.allclose(mcg_op, qiskit_op))

    def test_su2_iten(self):
        alpha = np.random.rand() + 1.j * np.random.rand()
        beta = np.random.rand() + 1.j * np.random.rand()

        for n_qubits in range(1, 10):
            self._su2_iten_compare(alpha, beta, n_qubits)
        
        for n_qubits in range(8, 15):
            self._su2_iten_count(alpha, beta, n_qubits)

    def test_su2_sec_diag_real(self):
        alpha = np.random.rand() + 1.j * np.random.rand()
        beta = np.random.rand()

        for n_qubits in range(1, 8):
            self._su2_compare(alpha, beta, n_qubits)
            self._su2_count(alpha, beta, n_qubits)

    def test_su2_pri_diag_real(self):
        alpha = np.random.rand()
        beta = np.random.rand() + 1.j * np.random.rand()

        for n_qubits in range(1, 10):
            self._su2_compare(alpha, beta, n_qubits)
            self._su2_count(alpha, beta, n_qubits)

    def test_su2_both_diag_real(self):
        alpha = np.random.rand()
        beta = np.random.rand()

        for n_qubits in range(1, 8):
            self._su2_compare(alpha, beta, n_qubits)
            self._su2_count(alpha, beta, n_qubits)

    def test_su2_both_diag_complex(self):
        alpha = np.random.rand() + 1.j * np.random.rand()
        beta = np.random.rand() + 1.j * np.random.rand()

        for n_qubits in range(1, 10):
            self._su2_compare(alpha, beta, n_qubits)
            self._su2_count(alpha, beta, n_qubits)
            #self._su2_depth(alpha, beta, n_qubits)

    def test_u2(self):
        unitary = unitary_group.rvs(2)

        for n_qubits in range(1, 10):
            self._u2_compare(unitary, n_qubits)
            self._u2_count(unitary, n_qubits)
