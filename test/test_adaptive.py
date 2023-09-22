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

""" Test bidirectional state preparation """

from unittest import TestCase
import random
from cmath import isclose
import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qclib.state_preparation import AdaptiveInitialize, BaaLowRankInitialize, BdspInitialize, MergeInitialize
from qclib.util import get_state
from .util import measurement

backend = AerSimulator()
SHOTS = 100000

class TestAdaptive(TestCase):
    """ Testing adaptative """

    @staticmethod
    def baa_experiment(state):
        """ Run bdsp experiment """
        print('BAA')
        opt_params = {'strategy': 'brute_force', 'use_low_rank': True}
        circuit = BaaLowRankInitialize(state, opt_params=opt_params).definition

        n_qubits = int(np.log2(len(state)))
        classical_register = ClassicalRegister(n_qubits)
        circuit.add_register(classical_register)

        return measurement(circuit, n_qubits, classical_register, backend, SHOTS)

    @staticmethod
    def bdsp_experiment(state, split=None):
        """ Run bdsp experiment """
        print('BDSP')
        opt_params = {'split': split}
        circuit = BdspInitialize(state, opt_params=opt_params).definition

        n_qubits = int(np.log2(len(state)))
        classical_register = ClassicalRegister(n_qubits)
        circuit.add_register(classical_register)

        return measurement(circuit, n_qubits, classical_register, backend, SHOTS)

    @staticmethod
    def adsp_experiment(state, split=None):
        """ Run adaptative experiment """
        opt_params = {'split': split, 'reset_ancillae': True}
        gate = AdaptiveInitialize(state, opt_params=opt_params)
        circuit = gate.definition
        n_qubits = gate.n_output
        classical_register = ClassicalRegister(n_qubits)
        circuit.add_register(classical_register)
        print(circuit.draw())
        return measurement(circuit, n_qubits, classical_register, backend, SHOTS, dense=False, patterns=state.keys())

    @staticmethod
    def dense_state(n_qubits):
        vector = np.random.rand(2**n_qubits) #+ np.random.rand(2**n_qubits) * 1j
        vector = vector / np.linalg.norm(vector)

        # rnd_index = random.sample(range(2**n_qubits), 2**(n_qubits))
        rnd_index = random.sample(range(2**n_qubits), 4)
        rnd_index.sort()
        vector = [a if i in rnd_index else 0.0 for i, a in enumerate(vector)]
        vector = [a if not isclose(a, 0.0) else 0.0 for a in vector]
        vector = vector / np.linalg.norm(vector)

        return vector

    @staticmethod
    def sparse_state(dense_vector):
        n_qubits = int(np.log2(len(dense_vector)))

        vector = {f"{i:0{n_qubits}b}" : a for i, a in enumerate(dense_vector) if not isclose(a, 0.0)}

        return vector

    def test_bottom_up(self):
        """ Testing adaptative """

        dense_vector = self.dense_state(4)
        vector = self.sparse_state(dense_vector)

        state = self.adsp_experiment(vector, 1)

        self.assertTrue(np.allclose(np.power(np.abs( [v for _, v in sorted(vector.items())] ), 2), state, rtol=1e-01, atol=0.005))

    def test_top_down(self):
        """ Testing adaptative """
        n_qubits = 12
        dense_vector = self.dense_state(n_qubits)
        sparse_vector = self.sparse_state(dense_vector)

        # ADAPTIVE
        opt_params = {'split': n_qubits, 'reset_ancillae': False, 'global_phase': True}
        circuit = QuantumCircuit(n_qubits)
        AdaptiveInitialize.initialize(circuit, sparse_vector, opt_params=opt_params)
        state = get_state(circuit)
        #tmp = circuit.decompose(reps=10)
        transpiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)
        print('ADSP:')
        print('cx', transpiled.count_ops().get('cx', 0))
        print('depth', transpiled.depth())

        # print(np.abs(dense_vector))
        # print(np.angle(dense_vector))
        # print(np.abs(state))
        # print(np.angle(state))

        self.assertTrue(np.allclose(dense_vector, state))

        # MERGE
        circuit = QuantumCircuit(n_qubits)
        MergeInitialize.initialize(circuit, sparse_vector)
        state = get_state(circuit)
        transpiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)
        print('MERGE:')
        print('cx', transpiled.count_ops().get('cx', 0))
        print('depth', transpiled.depth())

        self.assertTrue(np.allclose(dense_vector, state))

        # opt_params = {'strategy': 'brute_force', 'use_low_rank': True}
        # circuit = QuantumCircuit(n_qubits)
        # BaaLowRankInitialize.initialize(circuit, dense_vector, opt_params=opt_params)
        # state = get_state(circuit)

        # transpiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)
        # print('BAA:')
        # print('cx', transpiled.count_ops().get('cx', 0))
        # print('depth', transpiled.depth())

        # self.assertTrue(np.allclose(dense_vector, state))

        # circuit = QuantumCircuit(n_qubits)
        # TopDownInitialize.initialize(circuit, dense_vector)
        # state = get_state(circuit)
        # self.assertTrue(np.allclose(dense_vector, state))

        # tmp = circuit.decompose(reps=10)
        # print('Top down:')
        # print('cx', tmp.count_ops().get('cx',0))
        # print('depth', tmp.depth())


    def test_sublinear(self):
        """ Testing adaptative """
        dense_vector = self.dense_state(4)
        vector = self.sparse_state(dense_vector)

        state = TestAdaptive.adsp_experiment(vector)

        self.assertTrue(np.allclose(np.power(np.abs( [v for _, v in sorted(vector.items())] ), 2), state, rtol=1e-01, atol=0.005))

    def test_fixed(self):
        """ Testing adaptative """
        n_qubits = 4
        dense_vector = [0.0] * 2**n_qubits
        dense_vector[2] = np.sqrt(0.2)
        dense_vector[3] = np.sqrt(0.2)
        dense_vector[10] = np.sqrt(0.3)
        dense_vector[11] = np.sqrt(0.3)
        sparse_vector = self.sparse_state(dense_vector)

        # ADAPTIVE
        opt_params = {'split': n_qubits, 'reset_ancillae': False, 'global_phase': True}
        circuit = QuantumCircuit(n_qubits)
        AdaptiveInitialize.initialize(circuit, sparse_vector, opt_params=opt_params)
        state = get_state(circuit)
        transpiled = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)
        print('ADSP:')
        print('cx', transpiled.count_ops().get('cx', 0))
        print('depth', transpiled.depth())

        self.assertTrue(np.allclose(dense_vector, state))
