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

from qiskit.quantum_info import Statevector
from qclib.transform import Qhwt
from qclib.state_preparation import FrqiInitialize


# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestQhwt(TestCase):
    """ Testing qclib.transform.qhwt """

    def test_watermark(self):
        n_qubits = randint(4, 8)
        levels = 1

        state_vector1 = np.random.rand(2**(n_qubits-1))
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        # Creates the quantum circuit.
        circuit = FrqiInitialize(
            state_vector1,
            opt_params={'rescale': True}
        ).definition

        state1 = Statevector(circuit)

        # Transforms the initial state.
        qhwt = Qhwt(n_qubits, levels)
        circuit.append(qhwt, range(n_qubits))

        # Generates the watermark data.
        state_vector2 = np.random.rand(2**(n_qubits-1))
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)
        # Lists patterns with bit=0 at `positions`.
        def patterns_with_bit_0(n, positions):
            return [
                i for i in range(2 ** n) if all(
                    not (i & (1 << p))
                    for p in positions
                )
            ]
        # The area of the diagonal detail can be
        # selected by restricting the highest
        # qubits to |y_{n−1}>=1 and |x_{n−1}>=1.
        positions = [(n_qubits-1)//2-1, n_qubits-2]
        patterns = patterns_with_bit_0(n_qubits-1, positions)
        for pattern in patterns:
            state_vector2[pattern] = 0.0

        # Initializes the watermark.
        opt_params = {'init_index_register': False, 'rescale': True}
        watermark = FrqiInitialize(
            state_vector2,
            opt_params=opt_params
        )
        circuit.append(watermark, range(n_qubits))

        # Reverts the transform.
        circuit.append(qhwt.inverse(), range(n_qubits))

        state2 = Statevector(circuit)

        # Compares the obtained state with the expected state.
        # Despite the watermark, the vectors should be very similar,
        # but not equal.
        self.assertFalse(np.array_equal(state1, state2))
        self.assertTrue(np.allclose(state1, state2))
