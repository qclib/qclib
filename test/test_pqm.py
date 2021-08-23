from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qclib.memory.pqm import initialize as pqm
from qclib.state_preparation.default import initialize
from qclib.util import get_counts


class TestPQM(TestCase):

    @staticmethod
    def _run_pqm(is_classical_pattern):
        # pqm memory data
        data = [0, 1, 0, 0]
        data = data / np.linalg.norm(data)

        # initialize quantum registers
        memory = QuantumRegister(2, 'm')
        aux = QuantumRegister(1, 'c')
        output = ClassicalRegister(1)
        circ = QuantumCircuit(memory, aux, output)

        # initialize data
        init_gate = initialize(data)
        circ.append(init_gate, memory)

        # initialize input pattern
        bin_input = [1, 0]

        if (is_classical_pattern):
            pqm(circ, bin_input, memory, aux, is_classical_pattern=True) # run pqm recovery algorithm
        else:
            q_bin_input = QuantumRegister(2)
            circ.add_register(q_bin_input)

            # Pattern basis encoding
            for k, b in enumerate(bin_input):
                if (b == 1):
                    circ.x(q_bin_input[k])

            pqm(circ, q_bin_input, memory, aux, is_classical_pattern=False) # run pqm recovery algorithm

        # measure output and verify results
        circ.measure(aux, output)
        counts = get_counts(circ)
        
        return counts
        
    def test_classical_input(self):
        # classical input pattern
        counts = TestPQM._run_pqm(True)
        
        self.assertTrue(counts['0'] / 1024 == 1)

    def test_quantum_input(self):
        # quantum input pattern
        counts = TestPQM._run_pqm(False)
        
        self.assertTrue(counts['0'] / 1024 == 1)
        


