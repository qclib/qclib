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

""" Test uniform amplitude initialize """

from unittest import TestCase
import numpy as np
from qclib.state_preparation import FnPointsInitialize
from qclib.util import get_state


class TestVentura(TestCase):
    """Testing qclib.state_preparation.ventura.initialize"""

    def test_binary_function(self):
        """
        Testing a function (f(z) in {0, 1, ..., N-1}) with z in {0,1}^n.
        """
        # couples of input and output.
        n_qubits = 4
        function_io = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 11: 0, 12: 1, 13: 2, 14: 3, 15: 4}
        function_io = {
            f"{input_z:0{n_qubits}b}": output_s
            for input_z, output_s in function_io.items()
        }

        nnzero = len(function_io)
        max_denominator = max(function_io.values()) - 1

        opt_params = {"n_output_values": max_denominator}
        circuit = FnPointsInitialize(function_io, opt_params=opt_params).definition
        state = get_state(circuit)

        desired_state = []
        for k in [f"{i:0{n_qubits}b}" for i in range(2**n_qubits)]:
            # The amplitudes of modulus "1/sqrt(m)" will be "2 pi / N" radians
            # apart from each other on the complex plane.
            if k in function_io:
                amplitude = (
                    -1
                    / np.sqrt(nnzero)
                    * np.exp(function_io[k] * 1j * 2 * np.pi / max_denominator)
                )
                desired_state.append(amplitude)
            else:
                desired_state.append(0)

        # The data is encoded in the first 2^n amplitudes of the circuit state vector.
        self.assertTrue(np.allclose(desired_state, state[: 2**n_qubits]))
