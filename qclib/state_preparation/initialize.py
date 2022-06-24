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

from math import log2
from qiskit import QuantumCircuit
from qiskit.circuit.gate import Gate
import numpy as np


class Initialize(Gate):

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        pass

    def inverse(self):
        inverse_gate = self.copy()

        inverse_gate.definition = QuantumCircuit(
            *self.definition.qregs,
            *self.definition.cregs,
            global_phase=-self.definition.global_phase,
        )
        inverse_gate.definition._data = [
            (inst.inverse(), qargs, cargs) for inst, qargs, cargs in reversed(self._definition)
        ]

        return inverse_gate

    def _get_num_qubits(self, params):
        self.num_qubits = log2(len(params))
        if not self.num_qubits.is_integer():
            Exception("The number of amplitudes is not a power of 2")
        self.num_qubits = int(self.num_qubits)

    def validate_parameter(self, parameter):
        if isinstance(parameter, (int, float, complex)):
            return complex(parameter)
        elif isinstance(parameter, np.number):
            return complex(parameter.item())
        else:
            raise Exception(f"invalid param type {type(parameter)} for instruction  {self.name}")
