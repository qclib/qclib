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

""" Test divide-and-conquer state preparation """

from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister
from qiskit_aer import AerSimulator
from qclib.state_preparation import DcspInitialize
from qclib.util import measurement
# from .util import measurement


backend = AerSimulator()
SHOTS = 8192


class TestInitialize(TestCase):
    """ Testing divide-and-conquer state preparation """

    @staticmethod
    def dcsp_experiment(state):
        """ Run divide-and-conquer state preparation """
        circuit = DcspInitialize(state).definition

        n_qubits = int(np.log2(len(state)))
        classical_register = ClassicalRegister(n_qubits)
        circuit.add_register(classical_register)

        return measurement(circuit, n_qubits, classical_register, backend, SHOTS)

    def test_dcsp(self):
        """ Testing divide-and-conquer state preparation """
        vector = np.random.rand(16) + np.random.rand(16) * 1j
        vector = vector / np.linalg.norm(vector)

        state = TestInitialize.dcsp_experiment(vector)

        self.assertTrue(np.allclose(np.power(np.abs(vector), 2), state, rtol=1e-01, atol=0.005))
