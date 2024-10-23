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

""" Test qclib.transform.qhwt """

from unittest import TestCase

from random import randint

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qclib.transform import Qhwt

# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestQhwt(TestCase):
    """ Testing qclib.transform.qhwt """

    def test_simple(self):
        '''
        It only tests whether the execution causes a run-time error.
        '''
        n_qubits = randint(4, 8)
        levels = randint(1, n_qubits)

        state_vector = np.random.rand(2**n_qubits) + np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        # Creates the quantum circuit.
        circuit = QuantumCircuit(n_qubits)
        circuit.initialize(state_vector)

        qhwt = Qhwt(n_qubits, levels)
        circuit.append(qhwt, range(n_qubits))

        circuit.append(qhwt.inverse(), range(n_qubits))

        state = Statevector(circuit)

        # Compares the obtained state with the expected state.
        self.assertTrue(np.allclose(state_vector, state))
