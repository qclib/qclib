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

import numpy as np

from skimage import data, transform
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
from qclib.transform import Qhwt
from qclib.state_preparation import FrqiInitialize


# pylint: disable=maybe-no-member
# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestQhwt(TestCase):
    """ Testing qclib.transform.qhwt """

    def test_watermark(self):
        n_qubits = 7
        n_qubits_x = (n_qubits-1)//2
        n_qubits_y = (n_qubits-1)//2

        levels = 1

        # Image data.
        image1 = data.camera()
        image1 = transform.resize(image1, (2**n_qubits_x, 2**n_qubits_y))

        image2 = data.horse()
        image2 = transform.resize(image2, (2**(n_qubits_x-1), 2**(n_qubits_y-1)))

        # Creates the quantum circuit with the image data.
        state_vector1 = image1.reshape(-1)
        state_vector1 = state_vector1 / np.linalg.norm(state_vector1)

        circuit = FrqiInitialize(
            state_vector1,
            opt_params={
                'rescale': True,
                'simplify': False
            }
        ).definition

        # State vector before the watermarking.
        state1 = Statevector(circuit)

        # Transforms the initial state.
        qhwt_x = Qhwt(n_qubits_x, levels).definition
        qhwt_y = Qhwt(n_qubits_y, levels).definition

        qhwt = QuantumCircuit(n_qubits-1)
        qhwt.append(qhwt_x, range(n_qubits_x))
        qhwt.append(qhwt_y, range(n_qubits_x, n_qubits-1))

        circuit.append(qhwt, range(n_qubits-1))

        # Generates the watermark data.
        extended_image2 = np.zeros((2**n_qubits_x, 2**n_qubits_y,))
        extended_image2[2**(n_qubits_x-1):,2**(n_qubits_y-1):] = image2
        image2 = extended_image2

        state_vector2 = image2.reshape(-1)
        state_vector2 = state_vector2 / np.linalg.norm(state_vector2)

        # Initializes the watermark.
        opt_params = {
            'init_index_register': False,
            'rescale': True,
            'simplify': False
        }
        watermark = FrqiInitialize(
            state_vector2,
            opt_params=opt_params
        ).definition
        circuit.append(watermark, range(n_qubits))

        # Reverts the transform.
        circuit.append(qhwt.inverse(), range(n_qubits-1))

        # State vector after the watermarking.
        state2 = Statevector(circuit)

        # Compares the obtained state with the expected state.
        # Despite the watermark, the vectors should be very similar,
        # but not equal.
        self.assertFalse(np.array_equal(state1, state2))
        self.assertTrue(state_fidelity(state1, state2) > 0.99)
