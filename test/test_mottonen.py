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

""" Test state preparation """

from unittest import TestCase
import numpy as np
from qiskit import execute, Aer
from qclib.state_preparation.mottonen import initialize
from qclib.util import get_state

backend = Aer.get_backend('qasm_simulator')
SHOTS = 8192


class TestInitialize(TestCase):
    """ Testing qclib.state_preparation.mottonen """
    @staticmethod
    def measurement(circuit):
        """ get state preparation counts"""

        circuit.measure_all()
        job = execute(circuit, backend, shots=SHOTS, optimization_level=0)

        counts = job.result().get_counts(circuit)

        counts2 = {}
        for j in range(2**circuit.num_qubits):
            pattern = '{:0{}b}'.format(j, circuit.num_qubits)
            if pattern in counts:
                counts2[pattern] = counts[pattern]
            else:
                counts2[pattern] = 0.0

        return [value/SHOTS for (key, value) in counts2.items()]

    @staticmethod
    def mottonen_experiment(state):
        """ Creates circuit with Mottonen state preparation algorithm"""
        circuit = initialize(state)

        return TestInitialize.measurement(circuit)

    def test_mottonen_state(self):
        """ Testiong Mottonen state preparation """
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        circuit = initialize(vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(vector, state))

    def test_mottonen_measure(self):
        """ Testiong Mottonen state preparation with measurements"""
        vector = np.random.rand(32) + np.random.rand(32) * 1j
        vector = vector / np.linalg.norm(vector)

        state = TestInitialize.mottonen_experiment(vector)

        self.assertTrue(np.allclose( np.power(np.abs(vector), 2), state, rtol=1e-01, atol=0.005))
