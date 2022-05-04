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

""" Test blackbox state preparation """
from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit

from qclib.state_preparation.blackbox import BlackBoxInitialize
from qclib.util import get_state


class TestBlackbox(TestCase):
    """ Test blackbox state preparation """

    def test_blackbox(self):
        """ Run blackbox state preparation """
        initialize = BlackBoxInitialize.initialize

        state = np.random.rand(16) - 0.5 + (np.random.rand(16) - 0.5) * 1j
        state = state / np.linalg.norm(state)

        q_circuit = QuantumCircuit(5)
        initialize(q_circuit, state.tolist())

        out = get_state(q_circuit)
        out = out.reshape((len(out)//2, 2))
        out = out[:, 0]

        self.assertTrue(np.allclose(state, out, atol=0.02))
