from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qclib.pqm import pqm
from qclib.state_preparation import initialize
from qclib.util import get_counts


class TestPQM(TestCase):

    def test_pqm(self):
        # pqm memory data
        data = [1, 0, 0, 0]
        data = data / np.linalg.norm(data)

        # initialize quantum registers
        memory = QuantumRegister(2, 'm')
        aux = QuantumRegister(1, 'c')
        output = ClassicalRegister(1)
        circ = QuantumCircuit(memory, aux, output)

        # initialize data
        init_gate = initialize(data)
        circ.append(init_gate, memory)

        # run pqm recovery algorithm
        bin_input = [0, 0]
        pqm(circ, bin_input, memory, aux)

        # measure output and verify results
        circ.measure(aux, output)
        counts = get_counts(circ)

        self.assertTrue(counts['0'] / 1024 == 1)


