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
Tests for the gleining.py module.
"""

import unittest
import numpy as np
from qclib.state_preparation.gleining import initialize
from qclib.util import get_state, build_state_dict

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


class TestGleining(unittest.TestCase):

    def test_two_states_uniform(self):
        state_vector = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 1, 0, 0])
        state_dict = build_state_dict(state_vector)
        circ = initialize(state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_superposition(self):
        state_vector = 1 / np.sqrt(168) * np.array([0, 2, 0, 0, 8, 0, 0, 10])
        state_dict = build_state_dict(state_vector)
        circ = initialize(state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_uniform_superposition(self):
        state_vector = 1 / np.sqrt(3) * np.array([0, 1, 0, 0, 1, 0, 0, 1])
        state_dict = build_state_dict(state_vector)
        circ = initialize(state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_three_states_superposition_with_complex_features(self):
        state_vector = np.array([0, complex(np.sqrt(0.1), np.sqrt(0.1)), 0, 0,
                                 complex(np.sqrt(0.1), np.sqrt(0.2)), 0, 0, np.sqrt(0.5)])
        state_dict = build_state_dict(state_vector)
        circ = initialize(state_dict)
        state = get_state(circ)
        self.assertTrue(np.allclose(state_vector, state))

    def test_raises_error_input_not_dict(self):

        with self.assertRaises(Exception): 
            initialize([])
