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


class CVQRAM:
    """ https://arxiv.org/abs/2011.07977 """
    def __init__(self, nbits, data, mode='v-chain'):

        self.initialization(nbits, mode)
        norm = 1
        if mode == 'v-chain':
            self.circuit.x(self.qr_u2[0])
        elif mode == 'mct':
            self.circuit.x(self.qr_u0[1])

        control = range(self.memory.size)
        for binary_string, amplitude in data:

            self._load_binary(binary_string, mode)
            self.load_superposition(amplitude, mode, norm, control)
            self._load_binary(binary_string, mode)

    def initialization(self, nbits, mode):
        """ Initialize quantum registers"""

        self.nbits = nbits

        self.memory = QuantumRegister(self.nbits, name='m')


        if mode == 'mct':
            self.qr_u0 = QuantumRegister(2, name='u0')
            self.circuit = QuantumCircuit(self.qr_u0, self.memory)
        elif mode == 'v-chain':
            self.aux = QuantumRegister(nbits-1, name='anc')
            self.qr_u1 = QuantumRegister(1, name='u1')
            self.qr_u2 = QuantumRegister(1, name='u2')
            self.circuit = QuantumCircuit(self.qr_u1, self.qr_u2, self.memory, self.aux, )

    def _load_binary(self, binary_string, mode):

        for bit_index, bit in enumerate(binary_string):

            if bit == '1':
                if mode == 'v-chain':
                    self.circuit.cx(self.qr_u1[0], self.memory[bit_index])
                elif mode == 'mct':
                    self.circuit.cx(self.qr_u1[1], self.memory[bit_index])
            elif bit == '0':
                self.circuit.x(self.memory[bit_index])

    def load_superposition(self, feature, mode, norm, control):
        """
        Load pattern in superposition
        """

        alpha, beta, phi = _compute_matrix_angles(feature, norm)

        if mode == 'v-chain':

            self.circuit.rccx(self.memory[control[0]],
                              self.memory[control[1]], self.aux[0])

            for j in range(2, len(control)):
                self.circuit.rccx(self.memory[control[j]], self.aux[j - 2], self.aux[j - 1])

            self.circuit.cx(self.aux[len(control) - 2], self.qr_u1[0])

            self.circuit.cu3(alpha, beta, phi, self.qr_u1[0], self.qr_u2[0])

            self.circuit.cx(self.aux[len(control) - 2], self.qr_u1[0])

            for j in reversed(range(2, len(control))):
                self.circuit.rccx(self.memory[control[j]], self.aux[j - 2], self.aux[j - 1])

            self.circuit.rccx(self.memory[control[0]],
                              self.memory[control[1]], self.aux[0])

        if mode == 'mct':
            self.circuit.mct(self.memory, self.qr_u0[0])
            self.circuit.cu3(alpha, beta, phi, self.qr_u0[0], self.qr_u0[1])
            self.circuit.mct(self.memory, self.qr_u0[0])
        norm = norm - np.absolute(np.power(feature, 2))

def cvqram_initialize(state):
    """
    Creates a circuit to initialize a quantum state arXiv:2011.07977
    """
    qbit = state[0][0]
    size = len(qbit)
    n_qubits = int(size)
    memory = CVQRAM(n_qubits, state)
    return memory.circuit
