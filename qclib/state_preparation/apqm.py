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

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.util import _compute_matrix_angles

class CVQRAM:
    """ """
    def __init__(self, nbits, data, mode='v-chain'):

        self.initialization(nbits, mode)

        if mode == 'v-chain':
            #self.circuit.x(self.u1[0])
            self.circuit.x(self.u2[0])
        elif mode == 'mct':
            self.circuit.x(self.u0[1])

        for binary_string, amplitude in data:

            self._load_binary(binary_string)
            self._load_superposition(amplitude)
            self._load_binary(binary_string)

    def initialization(self, nbits, mode):
        self.mode = mode
        self.nbits = nbits
        self.norm = 1
        self.memory = QuantumRegister(self.nbits, name='m')
        self.control = range(self.memory.size)

        #self.circuit = QuantumCircuit(self.memory, self.u, self.aux)
        if self.mode=='mct':
            self.u0 = QuantumRegister(2, name='u0')
            self.circuit = QuantumCircuit(self.u0 ,self.memory)
        elif self.mode=='v-chain':
            self.aux = QuantumRegister(nbits-1, name='anc')
            self.u1 = QuantumRegister(1, name='u1')
            self.u2 = QuantumRegister(1, name='u2')
            self.circuit = QuantumCircuit(self.u1,  self.u2,  self.memory, self.aux,)


    # def mcxvchain(self, memory, anc, lst_ctrl, tgt):

    #     self.circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
    #     for j in range(2, len(lst_ctrl)):
    #         self.circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])

    #     self.circuit.cx(anc[len(lst_ctrl) - 2], tgt)#TODO mudar

    #     for j in reversed(range(2, len(lst_ctrl))):
    #         self.circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])
    #     self.circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])


    def _load_binary(self, binary_string):

        for bit_index, bit in enumerate(binary_string):

            if bit == '1':
                if self.mode=='v-chain':
                    self.circuit.cx(self.u1[0], self.memory[bit_index])
                elif self.mode=='mct':
                    self.circuit.cx(self.u1[1], self.memory[bit_index])
            elif bit == '0':
                self.circuit.x(self.memory[bit_index])


    # def flip_flop(self):
    #     for k in self.control:
    #         self.circuit.cx(self.u[0], self.memory[k])

    # @staticmethod
    # def select_controls(binary_string):
    #     control = []
    #     for k, bit in enumerate(binary_string[::-1]):
    #         if bit == '1':
    #             control.append(k)
    #     return control

    def _load_superposition(self, feature):
        """
        Load pattern in superposition
        """

        #alpha, beta, phi = _compute_matrix_angles(feature, self.norm)
        # gate.u3(0, alpha, beta, phi)
        alpha, beta, phi = _compute_matrix_angles(feature, self.norm)
        #U = U3Gate(alpha, beta, phi)
        # if self.mode == 'noancilla':
        #     custom = U3Gate(alpha, beta, phi).control(len(self.control))
        #     self.circuit.append(custom, self.memory[self.control] + [self.u0[0]])

        if self.mode =='v-chain':

            self.circuit.rccx(self.memory[self.control[0]],
                              self.memory[self.control[1]], self.aux[0])

            for j in range(2, len(self.control)):
                self.circuit.rccx(self.memory[self.control[j]], self.aux[j - 2], self.aux[j - 1])

            self.circuit.cx(self.aux[len(self.control) - 2], self.u1[0])

            self.circuit.cu3(alpha, beta, phi, self.u1[0], self.u2[0])

            self.circuit.cx(self.aux[len(self.control) - 2], self.u1[0])

            for j in reversed(range(2, len(self.control))):
                self.circuit.rccx(self.memory[self.control[j]], self.aux[j - 2], self.aux[j - 1])

            self.circuit.rccx(self.memory[self.control[0]],
                              self.memory[self.control[1]], self.aux[0])

        if self.mode =='mct':
            self.circuit.mct(self.memory, self.u0[0])
            self.circuit.cu3(alpha, beta, phi, self.u0[0], self.u0[1])
            self.circuit.mct(self.memory, self.u0[0])
        self.norm = self.norm - np.absolute(np.power(feature, 2))
        # self.circuit.cu3(alpha, beta, phi, self.aux[0], ancillae[1])



cvqram = CVQRAM

def cvqram_initialize(state):
    """
    Creates a circuit to initialize a quantum state arXiv:2011.07977
    """
    qbit = state[0][0]
    size = len(qbit)
    n_qubits = int(size)
    memory = CVQRAM(n_qubits, state)
    return memory.circuit
