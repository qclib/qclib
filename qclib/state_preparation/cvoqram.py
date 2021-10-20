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

""" cvo-qram """

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import UGate
from qclib.util import _compute_matrix_angles


class CVOQRAM:
    """ cvoqram """
    def __init__(self, nbits, data, with_aux=True):
        self.with_aux = with_aux
        self.initialization(nbits)
        self.circuit.x(self.aux[0])
        for k, binary_string_end_feature in enumerate(data):
            binary_string, feature = binary_string_end_feature

            self.control = CVOQRAM._select_controls(binary_string)
            self._flip_flop()
            self._load_superposition(feature, with_aux)
            if k<len(data)-1:
                self._flip_flop()
            else:
                break


    def initialization(self, nbits):
        """ Inicialize quantum registers """
        self.aux = QuantumRegister(1, name='u')
        self.memory = QuantumRegister(nbits, name='m')

        if self.with_aux:
            self.anc = QuantumRegister(nbits-1, name='anc')
            self.circuit = QuantumCircuit(self.aux, self.anc, self.memory)
        else:
            self.circuit = QuantumCircuit(self.aux, self.memory)
        self.nbits = nbits
        self.norm = 1


    def _flip_flop(self):

        for k in self.control:
            self.circuit.cx(self.aux[0], self.memory[k])

    @staticmethod
    def _select_controls(binary_string):
        control = []
        for k, bit in enumerate(binary_string[::-1]):
            if bit == 1:
                control.append(k)
        return control



    def mcuvchain(self, alpha, beta, phi):
        """
         N-qubit controlled-unitary gate
        """


        lst_ctrl = self.control
        lst_ctrl_reversed = list(reversed(lst_ctrl))
        self.circuit.rccx(self.memory[lst_ctrl_reversed [0]],
                          self.memory[lst_ctrl_reversed[1]],
                          self.anc[self.nbits-2])

        tof = {}
        i = self.nbits-1
        for ctrl in lst_ctrl_reversed [2:]:
            self.circuit.rccx(self.anc[i-1],
                              self.memory[ctrl],
                              self.anc[i-2])
            tof[ctrl] = [i-1, i-2]
            i-=1

        self.circuit.cu(alpha, beta, phi, 0, self.anc[i-1], self.aux[0])

        for ctrl in lst_ctrl[:-2]:
            self.circuit.rccx(self.anc[tof[ctrl][0]],
                              self.memory[ctrl],
                              self.anc[tof[ctrl][1]])

        self.circuit.rccx(self.memory[lst_ctrl[-1]],
                          self.memory[lst_ctrl[-2]],
                          self.anc[self.nbits-2])



    def _load_superposition(self, feature, with_aux=True):
        """
        Load pattern in superposition
        """

        theta, phi, lam = _compute_matrix_angles(feature, self.norm)

        if len(self.control) == 0:
            self.circuit.u(theta, phi, lam, self.aux[0])
        elif len(self.control) == 1:
            self.circuit.cu(theta, phi, lam, 0, self.memory[self.control[0]], self.aux[0])
        else:
            if with_aux:
                self.mcuvchain(theta, phi, lam)
            else:
                gate = UGate(theta, phi, lam).control(len(self.control))
                self.circuit.append(gate, self.memory[self.control] + [self.aux[0]])

        self.norm = self.norm - np.absolute(np.power(feature, 2))

def cvoqram_initialize(state, with_aux=True):
    """
    Creates a circuit to initialize a quantum state arXiv:

    For instance, to initialize the state a|001>+b|100>
        $ state = [('001', a), ('100', b)]
        $ circuit = sparse_initialize(state)

    Parameters
    ----------
    state: list of [(str,float)]
        A unit vector representing a quantum state.
        str: binary string
        float: amplitude

    Returns
    -------
    QuantumCircuit to initialize the state

    """
    qbit = state[0][0]
    size = len(qbit)
    n_qubits = int(size)
    memory = CVOQRAM(n_qubits, state, with_aux)
    return memory.circuit
