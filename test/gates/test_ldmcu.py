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
import qiskit
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit.library import MCXGate
import qclib.util
from qclib.gates.ldmcu import Ldmcu

class TestLinearToffoli(TestCase):
    """ Testing qclib.gate.toffoli """

    def test_controlled_gate(self):
        """ Testing multi controlled gate """

        gate_u = unitary_group.rvs(2)

        controls = QuantumRegister(4)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)
        circuit.x(0)
        circuit.x(1)
        circuit.x(2)
        circuit.x(3)

        Ldmcu.ldmcu(circuit, gate_u, controls, target)

        state = qclib.util.get_state(circuit)
        self.assertTrue(np.isclose(state[15], gate_u[0, 0]))
        self.assertTrue(np.isclose(state[31], gate_u[1, 0]))

        controls2 = QuantumRegister(4)
        target2 = QuantumRegister(1)
        circuit2 = QuantumCircuit(controls2, target2)
        circuit2.x(0)
        circuit2.x(1)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(4)

        Ldmcu.ldmcu(circuit2, gate_u, controls2, target2)

        state = qclib.util.get_state(circuit2)
        self.assertTrue(np.isclose(state[15], gate_u[0, 1]))
        self.assertTrue(np.isclose(state[31], gate_u[1, 1]))

    def test_linear_toffoli3(self):
        """ Testing Toffoli control 111"""
        gate_x = np.array([[0, 1], [1, 0]])

        controls = QuantumRegister(3)
        target = QuantumRegister(1)
        circuit = QuantumCircuit(controls, target)

        circuit.x(1)
        circuit.x(2)
        circuit.x(3)
        circuit.x(0)

        Ldmcu.ldmcu(circuit, gate_x, controls, target)

        state = qclib.util.get_state(circuit)
        exp_state = np.zeros(16, dtype=complex)
        exp_state[7] = 1

        self.assertTrue(np.allclose(state, exp_state))

    def test_linear_toffoli2(self):
        """ Testing Toffoli control 110"""
        gate_x = np.array([[0, 1], [1, 0]])

        controls2 = QuantumRegister(3)
        target2 = QuantumRegister(1)
        circuit2 = QuantumCircuit(controls2, target2)

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)
        state1 = qclib.util.get_state(circuit2)

        controls1 = QuantumRegister(3)
        target1 = QuantumRegister(1)
        circuit1 = qiskit.QuantumCircuit(controls1, target1)

        Ldmcu.ldmcu(circuit1, gate_x, controls1, target1, ctrl_state='110')

        circuit2.compose(circuit1, circuit2.qubits, inplace=True)
        state2 = qclib.util.get_state(circuit2)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli1(self):
        """ Testing Toffoli control 100"""
        gate_x = np.array([[0, 1], [1, 0]])

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(0)

        state1 = qclib.util.get_state(circuit2)

        controls = QuantumRegister(3)
        target = QuantumRegister(1)
        circuit = qiskit.QuantumCircuit(controls, target)

        Ldmcu.ldmcu(circuit, gate_x, controls, target, ctrl_state='100')
        circuit2.compose(circuit, circuit2.qubits, inplace=True)

        state2 = qclib.util.get_state(circuit2)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli0(self):
        """ Testing Toffoli control 000"""
        gate_x = np.array([[0, 1], [1, 0]])

        controls = QuantumRegister(3)
        target = QuantumRegister(1)
        mcgate_circuit = qiskit.QuantumCircuit(controls, target)

        Ldmcu.ldmcu(mcgate_circuit, gate_x, controls, target, ctrl_state="000")

        controls_2 = QuantumRegister(3)
        target_2 = QuantumRegister(1)
        qiskit_circuit = QuantumCircuit(controls_2, target_2)

        qiskit_circuit.append(MCXGate(len(controls_2), ctrl_state='000'),
                              [*controls_2, target_2])

        state_qiskit = qclib.util.get_state(qiskit_circuit)
        state_mcgate = qclib.util.get_state(mcgate_circuit)

        self.assertTrue(np.allclose(state_qiskit, state_mcgate))

    def test_mct_toffoli(self):
        """ compare qiskit.mct and toffoli depth with 7 qubits """
        gate_x = np.array([[0, 1], [1, 0]])
        qcirc1 = qiskit.QuantumCircuit(6)
        qcirc1.mct([0, 1, 2, 3, 4], 5)
        t_qcirc1 = qiskit.transpile(qcirc1, basis_gates=['u', 'cx'])

        controls = QuantumRegister(5)
        target = QuantumRegister(1)
        qcirc2 = qiskit.QuantumCircuit(controls, target)
        Ldmcu.ldmcu(qcirc2, gate_x, controls, target)

        t_qcirc2 = qiskit.transpile(qcirc2, basis_gates=['u', 'cx'])

        self.assertTrue(t_qcirc2.depth() < t_qcirc1.depth())
