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
Implements the state preparation
defined at https://arxiv.org/abs/1003.5760.
"""

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qclib.unitary import unitary
from qclib.gates.initialize import Initialize
import topdown


class SVDInitialize(Initialize):
    """
    https://arxiv.org/abs/1003.5760
    This class implements a state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.
        """

        self._name = "svd"
        self._get_num_qubits(params)

        self._label = label
        if label is None:
            self._label = "SP"

            if inverse:
                self._label = "SPdg"

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        """
        State preparation using Schmidt decomposition arXiv:1003.5760
        """
        state = np.copy(self.params)
        n_qubits = self.num_qubits
        odd = n_qubits % 2
        state.shape = (int(2 ** (n_qubits // 2)), int(2 ** (n_qubits // 2 + odd)))
        matrix_u, matrix_d, matrix_v = np.linalg.svd(state)
        matrix_d = matrix_d / np.linalg.norm(matrix_d)
        reg_a = QuantumRegister(n_qubits // 2 + odd)
        reg_b = QuantumRegister(n_qubits // 2)
        circuit = QuantumCircuit(reg_a, reg_b)
        if len(matrix_d) > 2:
            gate = SVDInitialize(matrix_d)
            circuit.append(gate, reg_b)
        else:
            gate = topdown.TopDownInitialize(matrix_d)
            circuit.append(gate, reg_b)

        for k in range(int(n_qubits // 2)):
            circuit.cx(reg_b[k], reg_a[k])

        # apply gate U to the first register
        gate_u = unitary(matrix_u)
        circuit.append(gate_u, reg_b)

        # apply gate V to the second register
        gate_v = unitary(matrix_v.T)
        circuit.append(gate_v, reg_a)

        return circuit

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        """
        Appends a SVDInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(SVDInitialize(state), q_circuit.qubits)
        else:
            q_circuit.append(SVDInitialize(state), qubits)
