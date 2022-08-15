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
    Tests for the bergholm state preparation
"""
from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit

from qclib.state_preparation import BergholmInitialize
from qclib.util import get_state

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class TestBergholmInitialize(TestCase):

  def test_three_qubit_state(self):

    state = np.random.rand(8) + np.random.rand(8) * 1j
    state = state / np.linalg.norm(state)
    
    initialize = BergholmInitialize.initialize
    circuit = QuantumCircuit(3)
    
    initialize(circuit, state.tolist())

    output_state = get_state(circuit)
 
    self.assertTrue(np.allclose(output_state, state))
