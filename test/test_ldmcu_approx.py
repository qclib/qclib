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

""" Test qclib.gate.mc_gate """

from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group
import random 
import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.circuit.library import MCXGate
import qclib.util
from qclib.gates.ldmcu import Ldmcu
from qclib.gates.ldmcu_approx import LdmcuApprox

class TestLinearU2(TestCase):
    
    def _get_result_unitary(self, unitary, n_ctrl_base):
        exponent = n_ctrl_base-1
        param = (2**exponent -1)/ 2**exponent
        values, vectors = np.linalg.eig(unitary)
        gate = np.power(values[0] + 0j, param) * vectors[:, [0]] @ vectors[:, [0]].conj().T
        gate = (
            gate
            + np.power(values[1] + 0j, param) * vectors[:, [1]] @ vectors[:, [1]].conj().T
        )
        return gate

    def test_get_num_qubits(self):
        error=0.1
        u1 = np.array([[0,1], [1,0]])
        ldmcu_approx = LdmcuApprox(u1, num_controls=10, error=error)
        self.assertTrue(ldmcu_approx._get_num_base_ctrl_qubits(u1,error), 4) 

    def test_mcz(self):
        error = 0.1
        u = np.array([[1,0], [0,-1]])
        ldmcu_approx_test = LdmcuApprox(u, num_controls=100, error=error)
        base_ctrl_qubits = ldmcu_approx_test._get_num_base_ctrl_qubits(u,error)

        controls = QuantumRegister(base_ctrl_qubits)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)
        N = len(controls)
        for i in range(N):
            circuit.x(i)
        LdmcuApprox.ldmcu_approx(circuit, u, controls, target, error)
        state = qclib.util.get_state(circuit)
        res_u = self._get_result_unitary(u, base_ctrl_qubits)
        self.assertTrue(np.isclose(state[2**N-1], res_u[0,0]))
        self.assertTrue(np.isclose(state[2**(N+1)-1], res_u[1,0]))

        controls = QuantumRegister(base_ctrl_qubits+2)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)
        N = len(controls)
        for i in range(N):
            circuit.x(i)
        LdmcuApprox.ldmcu_approx(circuit, u, controls, target, error)
        state = qclib.util.get_state(circuit)
        res_u = self._get_result_unitary(u, base_ctrl_qubits)
        self.assertTrue(np.isclose(state[2**N-1], res_u[0,0]))
        self.assertTrue(np.isclose(state[2**(N+1)-1], res_u[1,0]))

    def test_mcx(self):
        error = 0.1
        u = np.array([[0,1], [1,0]])
        ldmcu_approx_test = LdmcuApprox(u, num_controls=100, error=error)
        base_ctrl_qubits = ldmcu_approx_test._get_num_base_ctrl_qubits(u,error)

        controls = QuantumRegister(base_ctrl_qubits)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)
        N = len(controls)
        for i in range(N):
            circuit.x(i)
        LdmcuApprox.ldmcu_approx(circuit, u, controls, target, error)
        state = qclib.util.get_state(circuit)
        res_u = self._get_result_unitary(u, base_ctrl_qubits)
        self.assertTrue(np.isclose(state[2**N-1], res_u[0,0]))
        self.assertTrue(np.isclose(state[2**(N+1)-1], res_u[1,0]))

        controls = QuantumRegister(base_ctrl_qubits+2)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)
        N = len(controls)
        for i in range(N):
            circuit.x(i)
        LdmcuApprox.ldmcu_approx(circuit, u, controls, target, error)
        state = qclib.util.get_state(circuit)
        res_u = self._get_result_unitary(u, base_ctrl_qubits)
        self.assertTrue(np.isclose(state[2**N-1], res_u[0,0]))
        self.assertTrue(np.isclose(state[2**(N+1)-1], res_u[1,0]))

    def test_mcz_cnot_count(self):
        error = 0.01
        u = np.array([[1, 0], [0, -1]])
        ldmcu_approx_test = LdmcuApprox(u, num_controls=100, error=error)
        base_ctrl_qubits = ldmcu_approx_test._get_num_base_ctrl_qubits(u,error)
        for n_controls in range(base_ctrl_qubits+60, base_ctrl_qubits+62):
            controls = QuantumRegister(n_controls)
            target = QuantumRegister(1)
            circuit_approx = qiskit.QuantumCircuit(controls, target)
            LdmcuApprox.ldmcu_approx(circuit_approx, u, controls, target, error)
            cnot_approx = qclib.util.get_cnot_count(circ=circuit_approx)

            controls = QuantumRegister(n_controls)
            target = QuantumRegister(1)
            circuit_og = qiskit.QuantumCircuit(controls, target)
            Ldmcu.ldmcu(circuit_og, u, controls, target)
            cnot_og = qclib.util.get_cnot_count(circ=circuit_og)

            self.assertLessEqual(cnot_approx, cnot_og)
            


    def test_mcx_cnot_count(self):
        error = 0.01
        u = np.array([[0,1], [1,0]])
        ldmcu_approx_test = LdmcuApprox(u, num_controls=100, error=error)
        base_ctrl_qubits = ldmcu_approx_test._get_num_base_ctrl_qubits(u,error)
        for n_controls in range(base_ctrl_qubits+60, base_ctrl_qubits+62):
            controls = QuantumRegister(n_controls)
            target = QuantumRegister(1)
            circuit_approx = qiskit.QuantumCircuit(controls, target)
            LdmcuApprox.ldmcu_approx(circuit_approx, u, controls, target, error)
            cnot_approx = qclib.util.get_cnot_count(circ=circuit_approx)

            controls = QuantumRegister(n_controls)
            target = QuantumRegister(1)
            circuit_og = qiskit.QuantumCircuit(controls, target)
            Ldmcu.ldmcu(circuit_og, u, controls, target)
            cnot_og = qclib.util.get_cnot_count(circ=circuit_og)

            self.assertLessEqual(cnot_approx, cnot_og)

    def test_to_compare_ldmcu_and_ldmcu_approx(self):

        unitary = np.array([[0, 1], [1, 0]])
        error = 1 * 10e-4
        ldmcu_approx_test = LdmcuApprox(unitary, num_controls=100, error=error)
        base_ctrl_qubits = ldmcu_approx_test._get_num_base_ctrl_qubits(unitary, error)

        controls = QuantumRegister(base_ctrl_qubits)
        target = QuantumRegister(1)

        ldmcu_circ = QuantumCircuit(controls, target)
        Ldmcu.ldmcu(ldmcu_circ, unitary, controls, target)

        ldmcu_approx_circ = QuantumCircuit(controls, target)
        LdmcuApprox.ldmcu_approx(ldmcu_approx_circ, unitary, controls, target, error)

        ldmcu_op = Operator(ldmcu_circ).data
        ldmcu_approx_op = Operator(ldmcu_approx_circ).data

        # absolute(a - b) <= (atol + rtol * absolute(b)
        self.assertTrue(np.allclose(ldmcu_op, ldmcu_approx_op, atol=0.001))
