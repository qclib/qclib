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

"""
Tests for the mottonen.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, execute, Aer
from qclib.state_preparation.mottonen import initialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

backend = Aer.get_backend('qasm_simulator')
SHOTS = 8192

class TestInitialize(TestCase):
    @staticmethod
    def measurement(circuit, n_qubits, classical_reg):
        circuit.measure(list(range(n_qubits)), classical_reg)

        job = execute(circuit, backend, shots=SHOTS, optimization_level=3)

        counts = job.result().get_counts(circuit)
        sum_values = sum(counts.values())

        counts2 = {}
        for i in range(2**n_qubits):
            pattern = '{:0{}b}'.format(i, n_qubits)
            if pattern in counts:
                counts2[pattern] = counts[pattern]
            else:
                counts2[pattern] = 0.0

        return [ value/sum_values for (key, value) in counts2.items() ]

    @staticmethod
    def mottonen_experiment(state):
        circuit = initialize(state)

        n_qubits = int(np.log2(len(state)))
        classical_reg = ClassicalRegister(n_qubits)
        circuit.add_register(classical_reg)

        return TestInitialize.measurement(circuit, n_qubits, classical_reg)

    def test_mottonen_state_real(self):
        state_vector = np.random.rand(32)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_mottonen_state_complex(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit = initialize(state_vector)

        state = get_state(circuit)

        self.assertTrue(np.allclose(state_vector, state))

    def test_mottonen_measure(self):
        state_vector = np.random.rand(32) + np.random.rand(32) * 1j
        state_vector = state_vector / np.linalg.norm(state_vector)

        state = TestInitialize.mottonen_experiment(state_vector)

        self.assertTrue(np.allclose( np.power(np.abs(state_vector),2), state,
                        rtol=1e-01, atol=0.005))
