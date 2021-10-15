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

""" Test sparse state preparation """

from unittest import TestCase
import numpy as np
from qclib.state_preparation.default import initialize, sparse_initialize
from qclib.state_preparation.sparse_isometry import _pivoting
from qclib.util import get_state


class TestInitialize(TestCase):
    """ qclib.state_preparation.sparse_isometry """
    def test_initialize(self):
        """ Testing dense initialize """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circ = initialize(vector)
        state = get_state(circ)

        self.assertTrue(np.allclose(vector, state))

    def test_pivoting_3nonzero(self):
        """ Testing pivoting """
        vector = [-1/np.sqrt(4), 0, 0, 0, 0, 0, 0, 1/np.sqrt(4),
                  0, 0, 0, 1/np.sqrt(4), 0, 0, 0, 1/np.sqrt(4)]

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

        circuit, next_state = _pivoting(index_zero, index_nonzero, 2, state=next_state)
        circ.compose(circuit.reverse_bits(), circ.qubits, inplace=True)

        vector2 = get_state(circ)
        self.assertTrue(np.allclose(vector2[:3], [vector[0], vector[7], vector[11]]))

    def test_sparse_initialize(self):
        """ Testing sparse initialization """
        log_nnzero = 3
        n_qubits = 8
        vector = np.zeros(2**n_qubits)

        for _ in range(2**log_nnzero):
            index = np.random.randint(0, 2**n_qubits)
            while vector[index] != 0.0:
                index = np.random.randint(0, 2**n_qubits)
            vector[index] = np.random.rand()# + np.random.rand() * 1j

        vector = vector / np.linalg.norm(vector)

        vector2 = {}
        for index, value in enumerate(vector):
            if not np.isclose(value, 0.0):
                txt = '{0:0' + str(n_qubits) + 'b}'
                index_txt = txt.format(index)
                vector2[index_txt] = vector[index]

        circ = sparse_initialize(vector2)
        calc_vector = get_state(circ)

        self.assertTrue(np.allclose(vector, calc_vector))
