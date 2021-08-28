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
        
    def test_majority_1(self):
        counts = TestMajority._run_majority([0,0])
        
        self.assertTrue(counts['0'] / 1024 == 1)

    def test_majority_2(self):
        counts = TestMajority._run_majority([0,1])
        
        self.assertTrue(counts['1'] / 1024 == 1)
       
    def test_majority_3(self):
        counts = TestMajority._run_majority([1,1])
        
        self.assertTrue(counts['1'] / 1024 == 1)
        
    def test_majority_4(self):
        counts = TestMajority._run_majority([0,0,0,1])
        
        self.assertTrue(counts['0'] / 1024 == 1)
        
    def test_majority_5(self):
        counts = TestMajority._run_majority([0,1,0,1])
        
        self.assertTrue(counts['1'] / 1024 == 1)

    def test_majority_6(self):
        counts = TestMajority._run_majority([1,0,0,1])
        
        self.assertTrue(counts['1'] / 1024 == 1)

    def test_majority_7(self):
        counts = TestMajority._run_majority([0,1,0,0])
        
        self.assertTrue(counts['0'] / 1024 == 1)

