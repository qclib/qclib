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
Tests for the frqi.py module.
"""

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qclib.state_preparation import FrqiInitialize


# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestFrqi(TestCase):

    def test_initialize(self):
        n_qubits = 6

        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'method': 'ucr'}
        )

        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'method': 'mcg'}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        self.assertTrue(np.allclose(state1, state2))

    def test_cnot_count_fixed(self):
        n_qubits = 6

        state_vector = [0.0] * 2**n_qubits
        i = 0
        while i < 2**n_qubits:
            state_vector[i] = 1.0
            i = i + 2**(n_qubits//2)

        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'method': 'ucr'}
        )

        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'method': 'auto'}
        )

        t_circuit1 = transpile(
            circuit1.decompose(),
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit2 = transpile(
            circuit2.decompose(),
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 > n_cx2)

    def test_cnot_count_random(self):
        n_qubits = 6

        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'method': 'ucr'}
        )

        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'method': 'auto'}
        )

        t_circuit1 = transpile(
            circuit1.decompose(),
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit2 = transpile(
            circuit2.decompose(),
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 >= n_cx2)
