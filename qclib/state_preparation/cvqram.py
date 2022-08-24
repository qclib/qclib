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

""" https://arxiv.org/abs/2011.07977 """

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.util import _compute_matrix_angles
from qclib.gates.initialize_sparse import InitializeSparse

# pylint: disable=maybe-no-member


class CvqramInitialize(InitializeSparse):
    """
    Initializing the Amplitude Distribution of a Quantum State
    https://arxiv.org/abs/2011.07977

    This class implements a sparse state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        self._name = "cvqram"
        self._get_num_qubits(params)
        self.norm = 1

        default_mode = "v-chain"
        if opt_params is None:
            self.mode = default_mode
        else:
            if opt_params.get("mode") is None:
                self.mode = default_mode
            else:
                self.mode = opt_params.get("mode")

        if label is None:
            label = "CVSP"

        super().__init__(self._name, self.num_qubits, params.items(), label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        memory = QuantumRegister(self.num_qubits, name="m")

        if self.mode == "mct":
            aux = None
            qr_u1 = QuantumRegister(2, name="u1")
            circuit = QuantumCircuit(qr_u1, memory)
            circuit.x(qr_u1[1])
        elif self.mode == "v-chain":
            aux = QuantumRegister(self.num_qubits - 1, name="anc")
            qr_u1 = QuantumRegister(1, name="u1")
            qr_u2 = QuantumRegister(1, name="u2")
            circuit = QuantumCircuit(
                qr_u1,
                qr_u2,
                memory,
                aux,
            )
            circuit.x(qr_u2[0])

        self.norm = 1
        control = range(self.num_qubits)
        for binary_string, amplitude in self.params:
            self._load_binary(circuit, binary_string, self.mode, memory, qr_u1)
            self._load_superposition(
                circuit, amplitude, self.mode, control, memory, qr_u1, qr_u2, aux
            )
            self._load_binary(circuit, binary_string, self.mode, memory, qr_u1)

        return circuit

    @staticmethod
    def _load_binary(circuit, binary_string, mode, memory, qr_u1):
        for bit_index, bit in enumerate(binary_string):
            if bit == "1":
                if mode == "v-chain":
                    circuit.cx(qr_u1[0], memory[bit_index])
                elif mode == "mct":
                    circuit.cx(qr_u1[1], memory[bit_index])
            elif bit == "0":
                circuit.x(memory[bit_index])

    def _load_superposition(
        self, circuit, feature, mode, control, memory, qr_u1, qr_u2, aux
    ):
        """
        Load pattern in superposition
        """

        alpha, beta, phi = _compute_matrix_angles(feature, self.norm)

        if mode == "mct":
            circuit.mct(memory, qr_u1[0])
            circuit.cu3(alpha, beta, phi, qr_u1[0], qr_u1[1])
            circuit.mct(memory, qr_u1[0])
        elif mode == "v-chain":
            circuit.rccx(memory[control[0]], memory[control[1]], aux[0])

            for j in range(2, len(control)):
                circuit.rccx(memory[control[j]], aux[j - 2], aux[j - 1])

            circuit.cx(aux[len(control) - 2], qr_u1[0])
            circuit.cu3(alpha, beta, phi, qr_u1[0], qr_u2[0])
            circuit.cx(aux[len(control) - 2], qr_u1[0])

            for j in reversed(range(2, len(control))):
                circuit.rccx(memory[control[j]], aux[j - 2], aux[j - 1])

            circuit.rccx(memory[control[0]], memory[control[1]], aux[0])

        self.norm = self.norm - np.absolute(np.power(feature, 2))

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        if qubits is None:
            q_circuit.append(
                CvqramInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(CvqramInitialize(state, opt_params=opt_params), qubits)
