from unittest import TestCase

import qclib.util
from qclib.gate.linear_toffoli import toffoli
import qiskit




class TestLinearToffoli(TestCase):
    def test_linear_toffoli(self):
        circuit2 = qiskit.QuantumCircuit(3)
        circuit2.x(0)
        circuit2.x(1)
        circuit = toffoli()
        circuit = circuit2 + circuit
        circuit.measure_all()

        counts = qclib.util.get_counts(circuit)
        print(counts)
        self.assertTrue(counts['111'] == 1024)




