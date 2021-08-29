from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qclib.gate.majority import operate as majority
from qclib.util import get_counts
            
class TestMajority(TestCase):

    @staticmethod
    def _run_majority(bin_input):
        # initialize quantum registers
        controls = QuantumRegister(len(bin_input), 'controls')
        target = QuantumRegister(1, 'target')
        output = ClassicalRegister(1)
        circuit = QuantumCircuit(controls, target, output)

        # Pattern basis encoding
        for k, b in enumerate(bin_input):
            if (b == 1):
                circuit.x(controls[k])

        majority(circuit, controls, target)

        # measure output and verify results
        circuit.measure(target, output)
        counts = get_counts(circuit)
        
        return counts
    
    @staticmethod
    def _test_majority(self, n):
        for i in range(2**n):
            bits = [ int(j) for j in '{:0{}b}'.format(i, n) ]
            counts = TestMajority._run_majority(bits)
            
            if (sum(bits) >= n/2):
                self.assertTrue(counts['1'] / 1024 == 1)
            else:
                self.assertTrue(counts['0'] / 1024 == 1)
    
    # Keep tests separated by number of qubits to make it easier to identify errors.
    def test_majority_2(self):
        TestMajority._test_majority(self, 2)
        
    def test_majority_3(self):
        TestMajority._test_majority(self, 3)

    def test_majority_4(self):
        TestMajority._test_majority(self, 4)

    def test_majority_5(self):
        TestMajority._test_majority(self, 5)

    def test_majority_6(self):
        TestMajority._test_majority(self, 6)

    def test_majority_7(self):
        TestMajority._test_majority(self, 7)

    def test_majority_8(self):
        TestMajority._test_majority(self, 8)

    def test_majority_9(self):
        TestMajority._test_majority(self, 9)

