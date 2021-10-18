from unittest import TestCase

import numpy as np
import qclib.util
from qclib.gate.toffoli import toffoli
import qiskit




class TestLinearToffoli(TestCase):

    def test_linear_toffoli3(self):
        circuit2 = qiskit.QuantumCircuit(4)

        circuit2.x(1)
        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)

        circuit = qiskit.QuantumCircuit(4)
        toffoli(circuit, [1, 2,3], 0)
        circuit = circuit2 + circuit
        # circuit.measure_all()

        state = qclib.util.get_state(circuit)
        exp_state = np.zeros(16, dtype=complex)
        exp_state[14] = -1j
        print(state)
        self.assertTrue(np.allclose(state, exp_state))

    def test_linear_toffoli2(self):
        circuit2 = qiskit.QuantumCircuit(4)


        circuit2.x(2)
        circuit2.x(3)
        circuit2.x(0)
        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(circuit, [1, 2,3], 0)
        circuit = circuit2 + circuit
        # circuit.measure_all()

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli1(self):
        circuit2 = qiskit.QuantumCircuit(4)


        circuit2.x(2)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(circuit, [1, 2,3], 0)
        circuit = circuit2 + circuit
        # circuit.measure_all()

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))

    def test_linear_toffoli0(self):
        circuit2 = qiskit.QuantumCircuit(4)

        state1 = qclib.util.get_state(circuit2)

        circuit = qiskit.QuantumCircuit(4)

        toffoli(circuit, [1, 2,3], 0)
        circuit = circuit2 + circuit
        # circuit.measure_all()

        state2 = qclib.util.get_state(circuit)

        self.assertTrue(np.allclose(state1, state2))