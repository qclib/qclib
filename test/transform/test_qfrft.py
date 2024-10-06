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

""" Test qclib.transform.qfrft """

from unittest import TestCase

from random import randint, random

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit.circuit.library import QFT
from qclib.transform import Qfrft

# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestQfrft(TestCase):
    """ Testing qclib.transform.qfrft """

    def test_transformation(self):
        '''
        Verifies if the circuit produces the expected state by using a
        randomly chosen eigenvector of the Quantum Fourier Transform (QFT).
        In the context of the article, any state |u> is defined in terms
        of the eigenvectors |u_i> of the QFT.
        '''

        n_qubits = randint(4, 8)          # Size of the quantum state |u>.
        alpha = random()                  # F^alpha.
        index = randint(0, 2**n_qubits-1) # Eigenvector index.

        # Select an eigenvector as the quantum state |u>.
        qft_matrix = Operator(
            QFT(num_qubits=n_qubits, inverse=True)
        ).data

        e_vals, e_vecs = np.linalg.eig(qft_matrix)
        state_vector = e_vecs[:,index]

        # Creates the quantum circuit.
        init = QuantumCircuit(n_qubits)
        qfrft = Qfrft(n_qubits, alpha)

        init.initialize(state_vector)

        circuit = QuantumCircuit(n_qubits + 2)
        circuit.append(init, range(2, n_qubits+2))
        circuit.append(qfrft, range(n_qubits+2))

        # Obtains the state |u⟩ from the quantum circuit.
        full_state = Statevector(circuit)
        quantum_state = [
            v for k, v in full_state.to_dict().items() if k[-2:]=="00"
        ]

        # Calculates the expected state for comparison.
        phi = {1.0: 0, -1.0j: 1, -1.0: 2, 1.0j: 3} # See phi def. after eq. (25).
        expected_state = np.exp(                   # See eq. (29).
            -np.pi*1j*alpha*phi[np.round(e_vals[index])]/2
        ) * state_vector

        # Compares the obtained state with the expected state.
        self.assertTrue(np.allclose(quantum_state, expected_state))

    def test_aux_state(self):
        '''
        Checks if the probability amplitudes for control (auxiliary) qubit
        states other than |00⟩ are zero (null).
        '''

        n_qubits = randint(4, 8)
        alpha = random()
        state_vector = np.random.rand(2**n_qubits) + \
                       np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        init = QuantumCircuit(n_qubits)
        qfrft = Qfrft(n_qubits, alpha)

        init.initialize(state_vector)

        circuit = QuantumCircuit(n_qubits + 2)
        circuit.append(init, range(2, n_qubits+2))
        circuit.append(qfrft, range(n_qubits+2))

        state = Statevector(circuit)

        # Qiskit uses the little-endian convention, meaning that the first
        # two qubits in a quantum circuit, q0 and q1, are represented at the
        # end of the bitstring.
        null_amplitudes = [
            v for k, v in state.to_dict().items() if k[-2:]!="00"
        ]
        not_null_amplitudes = [
            v for k, v in state.to_dict().items() if k[-2:]=="00"
        ]

        null_zeros = [0.0] * len(null_amplitudes)
        not_null_zeros = [0.0] * len(not_null_amplitudes)

        self.assertTrue(np.allclose(null_zeros, null_amplitudes))
        self.assertFalse(np.allclose(not_null_zeros, not_null_amplitudes))
