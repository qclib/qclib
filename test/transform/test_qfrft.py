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

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qclib.util import get_cnot_count, get_depth
from qclib.transform import Qfrft


# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestQfrft(TestCase):
    """ Testing qclib.transform.qfrft """

    def test_aux_state(self):
        '''
        Checks if the probability amplitudes for control (auxiliary) qubit
        states other than |00⟩ are zero (null).
        '''

        n_qubits = 6
        state_vector = np.random.rand(2**n_qubits) + \
                       np.random.rand(2**n_qubits) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        init = QuantumCircuit(n_qubits)
        qfrft = Qfrft(n_qubits, 0.5)

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
