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

from math import log2, isclose
from qiskit.circuit.gate import Gate
import numpy as np


class Initialize(Gate):
    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        pass

    def inverse(self):
        inverse_gate = self.copy()

        inverse_gate.definition = self.definition.inverse()
        inverse_gate.label += "_dg"

        return inverse_gate

    def _get_num_qubits(self, params):
        self.num_qubits = log2(len(params))

        # Check if param is a power of 2
        if self.num_qubits == 0 or not self.num_qubits.is_integer():
            raise ValueError("The length of the state vector is not a positive power of 2.")

        # Check if probabilities (amplitudes squared) sum to 1
        if not isclose(sum(np.absolute(params) ** 2), 1.0, abs_tol=1e-10):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        self.num_qubits = int(self.num_qubits)

    def validate_parameter(self, parameter):
        if isinstance(parameter, (int, float, complex)):
            return complex(parameter)
        elif isinstance(parameter, np.number):
            return complex(parameter.item())
        else:
            raise TypeError(
                f"invalid param type {type(parameter)} for instruction {self.name}."
            )
