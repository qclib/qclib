
""" Test qclib.gate.toffoli """

from unittest import TestCase

import numpy as np
import qiskit
import qclib.util
from qclib.gate.toffoli import toffoli

class TestLinearToffoli(TestCase):
    """ Testing qclib.gate.toffoli """

    def test_linear_toffoli3(self):
        """ Testing Toffoli control 111"""

        circuit2 = qiskit.QuantumCircuit(4)

        circuit2.x(1)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)

        circuit = qiskit.QuantumCircuit(4)
        toffoli(circuit, [1, 2, 3], 0)
        circuit = circuit2 + circuit

        state = qclib.util.get_state(circuit)
        exp_state = np.zeros(16, dtype=complex)
        exp_state[14] = 1

        self.assertTrue(np.allclose(state, exp_state))

    def test_linear_toffoli2(self):
        """ Testing Toffoli control 110"""

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)
        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)
        toffoli(circuit, [3, 2, 1], 0)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli1(self):
        """ Testing Toffoli control 100"""

        circuit2 = qiskit.QuantumCircuit(4)
        circuit2.x(2)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(circuit, [0, 1, 2], 3)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli0(self):
        """ Testing Toffoli control 000"""

        circuit2 = qiskit.QuantumCircuit(4)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(circuit, [1, 2, 3], 0)
        circuit = circuit2 + circuit

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_mct_toffoli(self):
        """ compare qiskit.mct and toffoli depth with 7 qubits """

        qcirc1 = qiskit.QuantumCircuit(7)
        qcirc1.mct([0, 1, 2, 3, 4, 5], 6)
        t_qcirc1 = qiskit.transpile(qcirc1, basis_gates=['u', 'cx'])

        qcirc2 =  qiskit.QuantumCircuit(7)
        toffoli(qcirc2,[0, 1, 2, 3, 4, 5], 6)
        t_qcirc2 = qiskit.transpile(qcirc2, basis_gates=['u', 'cx'])

        self.assertTrue(t_qcirc2.depth() < t_qcirc1.depth())
