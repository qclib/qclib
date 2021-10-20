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
from qiskit import ClassicalRegister, execute, Aer
from qclib.state_preparation.dcsp import initialize

backend = Aer.get_backend('qasm_simulator')
SHOTS = 8192

class TestInitialize(TestCase):
    """ Testing divide-and-conquer state preparation """

    @staticmethod
    def measurement(circuit, n_qubits, classical_register):
        """ run circuit """
        circuit.measure(list(range(n_qubits)), classical_register)

        job = execute(circuit, backend, shots=SHOTS, optimization_level=3)

        counts = job.result().get_counts(circuit)

        counts2 = {}
        for k in range(2 ** n_qubits):
            pattern = '{:0{}b}'.format(k, n_qubits)
            if pattern in counts:
                counts2[pattern] = counts[pattern]
            else:
                counts2[pattern] = 0.0

        return [value/SHOTS for (key, value) in counts2.items()]

    @staticmethod
    def dcsp_experiment(state):
        """ Run divide-and-conquer state preparation """
        circuit = initialize(state)

        n_qubits = int(np.log2(len(state)))
        classical_register = ClassicalRegister(n_qubits)
        circuit.add_register(classical_register)

        return TestInitialize.measurement(circuit, n_qubits, classical_register)

    def test_dcsp(self):
        """ Testing divide-and-conquer state preparation """
        vector = np.random.rand(16) + np.random.rand(16) * 1j
        vector = vector / np.linalg.norm(vector)

        state = TestInitialize.dcsp_experiment(vector)

        self.assertTrue(np.allclose( np.power(np.abs(vector),2), state, rtol=1e-01, atol=0.005))
