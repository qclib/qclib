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
Tests for the merge.py module.
"""

import unittest
import numpy as np
from qiskit import QuantumCircuit
from qclib.state_preparation import MergeInitialize
from qclib.util import get_state, build_state_dict
from qiskit import transpile
from scipy.sparse import random

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestMergeInitialize(unittest.TestCase):
    """ Teste MergeInitialize Gate"""

    def test_two_states_uniform(self):
        state_vector = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 1, 0, 0])
        state_dict = build_state_dict(state_vector)
        initialize = MergeInitialize.initialize
        circ = QuantumCircuit(3)
        initialize(circ, state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_superposition(self):
        state_vector = 1 / np.sqrt(168) * np.array([0, 2, 0, 0, 8, 0, 0, 10])
        state_dict = build_state_dict(state_vector)
        initialize = MergeInitialize.initialize
        circ = QuantumCircuit(3)
        initialize(circ, state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_uniform_superposition(self):
        state_vector = 1 / np.sqrt(3) * np.array([0, 1, 0, 0, 1, 0, 0, 1])
        state_dict = build_state_dict(state_vector)
        initialize = MergeInitialize.initialize
        circ = QuantumCircuit(3)
        initialize(circ, state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_superposition_with_complex_features(self):
        state_vector = np.array([0, complex(np.sqrt(0.1), np.sqrt(0.1)), 0, 0,
                                 complex(np.sqrt(0.1), np.sqrt(0.2)), 0, 0, np.sqrt(0.5)])
        state_dict = build_state_dict(state_vector)
        initialize = MergeInitialize.initialize
        circ = QuantumCircuit(3)
        initialize(circ, state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_raises_error_input_not_dict(self):
        initialize = MergeInitialize.initialize
        circ = QuantumCircuit()
        with self.assertRaises(Exception): 
            initialize(circ, [])

    def test_8qb_sparse(self):
        state_dict = {
            '01100000': 0.11496980229422502, 
            '10010000': 0.2012068017595738,
            '11110000': 0.2552406117427385,
            '11001000': 0.24483730989689545,
            '00011000': 0.08064396530053637, 
            '01111000': 0.06609205232425505,
            '10000100': 0.2567251902135311,
            '00100010': 0.279179786457501,
            '11110010': 0.14972323818181424,
            '00000110': 0.054570286103576615,  
            '10101110': 0.1953959409811345,
            '00011110': 0.2476316976925628,
            '00111110': 0.2460713287965397,
            '00010001': 0.2880964575493704,
            '00011001': 0.11697558413298771,  
            '11100101': 0.15657582325155645,
            '00101101': 0.05059343291713247,
            '10011101': 0.21260965910383026,
            '11100011': 0.16144719592639006,
            '01110011': 0.24224885089395568,
            '10011011': 0.07542653172823867,
            '01111011': 0.2840232568261471, 
            '00100111': 0.2719803407586484,
            '01100111': 0.14940066988379283,  
            '11010111': 0.2025293455684502, 
            '01001111': 0.06045929196877916
        }
        initialize = MergeInitialize.initialize
        qc = QuantumCircuit(8)
        initialize(qc, state_dict)

        t_circuit = transpile(qc, basis_gates=['cx', 'u'])

    def test_several_qubit_sizes(self):
        for n_qubits in range(4, 12):
            state_vector = random(1, 2 ** n_qubits, density=0.1, random_state=42).A.reshape(-1)
            state_vector = state_vector / np.linalg.norm(state_vector)
            state_dict = build_state_dict(state_vector)

            initialize = MergeInitialize.initialize
            qc = QuantumCircuit(n_qubits)
            initialize(qc, state_dict)

            state = get_state(qc)
            self.assertTrue(np.allclose(state_vector, state))
