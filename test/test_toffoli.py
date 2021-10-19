
""" Test qclib.gate.toffoli """

from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group
import qiskit
import qclib.util
from qclib.gate.toffoli import toffoli


class TestLinearToffoli(TestCase):
    """ Testing qclib.gate.toffoli """

    def test_controlled_gate(self):

        gate_u = unitary_group.rvs(2)

        circuit = qiskit.QuantumCircuit(5)
        circuit.x(1)
        circuit.x(2)
        circuit.x(3)
        circuit.x(4)

        toffoli(gate_u, circuit, [4, 3, 2, 1], 0)

        state = qclib.util.get_state(circuit)
        self.assertTrue(np.isclose(state[30], gate_u[0,0]))
        self.assertTrue(np.isclose(state[31], gate_u[1, 0]))

        circuit2 = qiskit.QuantumCircuit(5)
        circuit2.x(0)
        circuit2.x(1)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(4)

        toffoli(gate_u, circuit2, [4, 3, 2, 1], 0)

        state = qclib.util.get_state(circuit2)
        self.assertTrue(np.isclose(state[30], gate_u[0, 1]))
        self.assertTrue(np.isclose(state[31], gate_u[1, 1]))

    def test_linear_toffoli3(self):
        """ Testing Toffoli control 111"""
        gate_x = np.array([[0, 1], [1, 0]])
        circuit = qiskit.QuantumCircuit(4)

        circuit.x(1)
        circuit.x(2)
        circuit.x(3)
        circuit.x(0)

        toffoli(gate_x, circuit, [3, 2, 1], 0)

        state = qclib.util.get_state(circuit)
        exp_state = np.zeros(16, dtype=complex)
        exp_state[7] = 1

        self.assertTrue(np.allclose(state, exp_state))

    def test_linear_toffoli2(self):
        """ Testing Toffoli control 110"""
        gate_x = np.array([[0, 1], [1, 0]])

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)
        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)
        toffoli(gate_x, circuit, [3, 2, 1], 0)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli1(self):
        """ Testing Toffoli control 100"""
        gate_x = np.array([[0, 1], [1, 0]])

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(2)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(gate_x, circuit, [0, 1, 2], 3)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli0(self):
        """ Testing Toffoli control 000"""
        gate_x = np.array([[0, 1], [1, 0]])

        circuit2 = qiskit.QuantumCircuit(4)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(gate_x, circuit, [1, 2, 3], 0)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_mct_toffoli(self):
        """ compare qiskit.mct and toffoli depth with 7 qubits """
        gate_x = np.array([[0, 1], [1, 0]])
        qcirc1 = qiskit.QuantumCircuit(6)
        qcirc1.mct([0, 1, 2, 3, 4], 5)
        t_qcirc1 = qiskit.transpile(qcirc1, basis_gates=['u', 'cx'])

        qcirc2 = qiskit.QuantumCircuit(6)
        toffoli(gate_x, qcirc2, [0, 1, 2, 3, 4], 5)
        t_qcirc2 = qiskit.transpile(qcirc2, basis_gates=['u', 'cx'])

        self.assertTrue(t_qcirc2.depth() < t_qcirc1.depth())
