from unittest import TestCase
import numpy as np
from qclib.state_preparation import initialize, _pivoting, sparse_initialize
from qclib.util import get_state
from qiskit import transpile


class TestInitialize(TestCase):
    def test_initialize(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circ = initialize(a)

        state = get_state(circ)

        self.assertTrue(np.allclose(a, state))

    def test_pivoting_2nonzero(self):
        vector = [np.sqrt(2)/np.sqrt(3), 0, 0, 0, 0, 0, 0, 1/np.sqrt(3), 0, 0, 0, 0, 0, 0, 0, 0]

        circ = initialize(vector)
        n_qubits = int(np.log2(len(vector)))
        txt = "{0:0" + str(n_qubits) + "b}"
        index_zero = txt.format(1)
        index_nonzero = txt.format(7)

        circuit, next_state = _pivoting(index_zero, index_nonzero, 2, state=vector)
        circ.compose(circuit, circ.qubits, inplace=True)
        vector2 = get_state(circ)
        self.assertTrue(np.isclose(vector2[0], vector[0]))
        self.assertTrue(np.isclose(vector2[1], vector[7]))

    def test_pivoting_3nonzero(self):
        vector = [-1/np.sqrt(4), 0, 0, 0, 0, 0, 0, 1/np.sqrt(4), 0, 0, 0, 1/np.sqrt(4), 0, 0, 0, 1/np.sqrt(4)]
        vector2 = {}

        for k, value in enumerate(vector):
            if value != 0:
                index = format(k, '04b')
                vector2[index] = value

        circ = initialize(vector)
        n_qubits = int(np.log2(len(vector)))
        txt = "{0:0" + str(n_qubits) + "b}"
        index_zero = txt.format(1)
        index_nonzero = txt.format(7)

        circuit, next_state = _pivoting(index_zero, index_nonzero, 2, state=vector2)

        circ.compose(circuit.reverse_bits(), circ.qubits, inplace=True)

        index_zero = txt.format(2)
        index_nonzero = txt.format(9)
        print([int(key,2) for key, value in next_state.items()])
        vector3 = get_state(circ)
        print(vector3.nonzero())
        circuit, next_state = _pivoting(index_zero, index_nonzero, 2, state=next_state)
        circ.compose(circuit.reverse_bits(), circ.qubits, inplace=True)

        # circuit, next_state = pivoting(index_zero, index_nonzero, 2, state=next_state)
        # circ.compose(circuit, circ.qubits, inplace=True)

        vector2 = get_state(circ)
        self.assertTrue(np.allclose(vector2, vector))


    def test_sparse_initialize(self):
        s = 2
        n = 8
        vector = np.zeros(2**n)

        for k in range(2**s):
            index = np.random.randint(0, 2**n)
            while vector[index] != 0.0:
                index = np.random.randint(0, 2**n)
            vector[index] = np.random.rand()# + np.random.rand() * 1j

        vector = vector / np.linalg.norm(vector)
        # vector = np.loadtxt('vetor')
        vector2 = {}
        for index, value in enumerate(vector):
            if not np.isclose(value, 0.0):
                txt = '{0:0' + str(n) + 'b}'
                index_txt = txt.format(index)
                vector2[index_txt] = vector[index]

        # vector = [1 / np.sqrt(4), 1 / np.sqrt(4), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 / np.sqrt(4), 0, 0, 1 / np.sqrt(4)]
        circ = sparse_initialize(vector2)
        calc_vector = get_state(circ)
        circt = transpile(circ, basis_gates=['u', 'cx'], optimization_level=3)
        print(circ.count_ops())
        print(circt.count_ops())
        print(circ.draw())
        self.assertTrue(np.allclose(vector, calc_vector))








