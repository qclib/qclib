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

    def test_simplify_auto(self):
        n_qubits = 6
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
            opt_params={'rescale':True, 'simplify': True}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False}
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

    def test_simplify_ucr(self):
        n_qubits = 6
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
            opt_params={'rescale':True, 'simplify': True, 'method': 'ucr'}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'method': 'ucr'}
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

    def test_simplify_mcg(self):
        n_qubits = 6
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
            opt_params={'rescale':True, 'simplify': True, 'method': 'mcg'}
        )
        circuit2 = QuantumCircuit(n_qubits+1)
        FrqiInitialize.initialize(
            circuit2,
            state_vector,
            opt_params={'rescale':True, 'simplify': False, 'method': 'mcg'}
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
        n_qubits = 6
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
            opt_params={'rescale':True}
        )

        pm = generate_preset_pass_manager(
            basis_gates=['u', 'cx'],
            optimization_level=0
        )
        t_circuit1 = pm.run(circuit1.decompose())

        n_cx1 = t_circuit1.count_ops()['cx']

        self.assertTrue(n_cx1 == 2**n_qubits // div)

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
        """
        See Figure 9 of the article:
        https://link.springer.com/article/10.1007/s11128-010-0177-y
        
        In it, theta_2 repeats three times, allowing for a further
        simplification, which consists of ignoring the multicontrolled
        operations of the angles that repeat, adding a Ry rotation of that
        angle, and subtracting it from the angles left with controls. With this
        simplification, only the multicontrolled operation theta_1 remains,
        with three controls, resulting in just ``n_cx2=8`` CNOTs
        (for ``'method': 'auto'``).
        """
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

        self.assertTrue(n_cx1 > n_cx2)
        self.assertTrue(np.allclose(state1, state2))

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

        self.assertTrue(n_cx1 >= n_cx2)
        self.assertTrue(np.allclose(state1, state2))
