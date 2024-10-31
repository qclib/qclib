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
from qiskit import QuantumCircuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.quantum_info import Statevector
from qclib.state_preparation import FrqiInitialize


# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestFrqi(TestCase):

    def test_simplify_1(self):
        '''
        simplify  | separability
        ------------------------
        True      | True
        False     | True
        '''
        n_qubits = 8
        div = 2**4

        pattern_vector = np.random.rand(2**n_qubits//div)
        state_vector = pattern_vector.copy()
        for _ in range(1, div):
            state_vector = np.concatenate((state_vector, pattern_vector,))
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'simplify': True, 'separability': True}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'separability': True}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 < n_cx2)
        self.assertTrue(np.allclose(state1, state2))

    def test_simplify_2(self):
        '''
        simplify  | separability
        ------------------------
        True      | False
        False     | False
        '''
        n_qubits = 8
        div = 2**4

        pattern_vector = np.random.rand(2**n_qubits//div)
        state_vector = pattern_vector.copy()
        for _ in range(1, div):
            state_vector = np.concatenate((state_vector, pattern_vector,))
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'simplify': True, 'separability': False}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'separability': False}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 <= n_cx2)
        self.assertTrue(np.allclose(state1, state2))

    def test_simplify_3(self):
        '''
        simplify  | separability
        ------------------------
        True      | True
        False     | False
        '''
        n_qubits = 8
        div = 2**4

        pattern_vector = np.random.rand(2**n_qubits//div)
        state_vector = pattern_vector.copy()
        for _ in range(1, div):
            state_vector = np.concatenate((state_vector, pattern_vector,))
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'simplify': True, 'separability': True}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'separability': False}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 < n_cx2)
        self.assertTrue(np.allclose(state1, state2))

    def test_simplify_4(self):
        n_qubits = 8
        div = 2**4

        pattern_vector = np.random.rand(2**n_qubits//div)
        state_vector = pattern_vector.copy()
        for _ in range(1, div):
            state_vector = np.concatenate((state_vector, pattern_vector,))
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'simplify': True, 'separability': False}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'separability': True}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 <= n_cx2)
        self.assertTrue(np.allclose(state1, state2))

    def test_separability(self):
        n_qubits = 8
        div = 2**4

        pattern_vector = np.random.rand(2**n_qubits//div)
        state_vector = pattern_vector.copy()
        for _ in range(1, div):
            state_vector = np.concatenate((state_vector, pattern_vector,))
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'separability': True}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'separability': False}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 < n_cx2)
        self.assertTrue(np.allclose(state1, state2))

    def test_initialize_simplify(self):
        n_qubits = 6

        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'method': 'ucr', 'simplify': True}
        )

        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'method': 'mcg', 'simplify': True}
        )

        state1 = Statevector(circuit1).data
        state2 = Statevector(circuit2).data

        self.assertTrue(np.allclose(state1, state2))

    def test_initialize(self):
        n_qubits = 6

        state_vector = np.random.rand(2**n_qubits)
        state_vector = state_vector / np.linalg.norm(state_vector)

        circuit1 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit1,
            state_vector,
            opt_params={'rescale':True, 'method': 'ucr', 'simplify': False}
        )

        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'method': 'mcg', 'simplify': False}
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

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

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

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())
        t_circuit2 = pm.run(circuit2.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']
        n_cx2 = t_circuit2.count_ops()['cx']

        self.assertTrue(n_cx1 >= n_cx2)
