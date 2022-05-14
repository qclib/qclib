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

""" sparse state preparation """

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qclib.state_preparation.lowrank import LRInitialize

# pylint: disable=maybe-no-member

class PivotStatePreparation:
    """ Pivot State Preparation """
    def __init__(self, state, aux=False):
        self.aux = aux
        self.n_qubits = len(list(state.keys())[0])
        self.n_qubits = int(self.n_qubits)

        non_zero = len(state)
        self.target_size = np.log2(non_zero)
        self.target_size = np.ceil(self.target_size)
        self.target_size = int(self.target_size)

        self.memory = qiskit.QuantumRegister(self.n_qubits, name='q')
        if self.aux:
            remain = list(range(self.n_qubits - self.target_size, self.n_qubits))
            n_anci = len(remain)

            anc = qiskit.QuantumRegister(n_anci - 1, name='anc')
            pivot_circuit = qiskit.QuantumCircuit(anc, self.memory)

        else:
            pivot_circuit = qiskit.QuantumCircuit(self.memory)

        self.next_state = state.copy()
        index_nonzero = self._get_index_nz(self.n_qubits - self.target_size)

        while index_nonzero is not None:
            self.index_zero = self._get_index_zero(non_zero, self.next_state)
            circ = self.pivoting(index_nonzero, self.target_size, aux)

            pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)

            index_nonzero = self._get_index_nz(self.n_qubits - self.target_size)

        dense_state = np.zeros(2 ** self.target_size, dtype=complex)
        for key, value in self.next_state.items():
            dense_state[int(key, 2)] = value

        if non_zero <= 2:
            initialize_circ = qiskit.QuantumCircuit(1)
            initialize_circ.initialize(dense_state)
        else:
            initialize_circ = QuantumCircuit(self.target_size)
            LRInitialize.initialize(initialize_circ, dense_state)

        if self.aux:
            self.sp_circuit = qiskit.QuantumCircuit(anc, self.memory)
            nun_aux = n_anci - 1
            self.sp_circuit.compose(
                initialize_circ,
                self.sp_circuit.qubits[nun_aux:nun_aux + self.target_size],
                inplace=True)

            self.sp_circuit.barrier()
            self.sp_circuit.compose(pivot_circuit.reverse_bits().reverse_ops(), inplace=True)
        else:
            self.sp_circuit = qiskit.QuantumCircuit(self.n_qubits)
            self.sp_circuit.compose(
                initialize_circ,
                self.sp_circuit.qubits[:self.target_size], inplace=True)

            self.sp_circuit.compose(
                pivot_circuit.reverse_bits().reverse_ops(),
                self.sp_circuit.qubits,
                inplace=True)

    def _next_state(self, ctrl_state, index_differ, remain, target_cx):
        tab = {'0': '1', '1': '0'}
        new_state = {}
        for index, _ in self.next_state.items():

            if index[index_differ] == ctrl_state:

                n_index = ''
                for k, _ in enumerate(index):
                    if k in target_cx:
                        n_index = n_index + tab[index[k]]
                    else:
                        n_index = n_index + index[k]

            else:
                n_index = index

            if n_index[remain[0]:] == self.index_zero[remain[0]:]:
                n_index = n_index[:index_differ] + \
                          tab[index[index_differ]] + n_index[index_differ + 1:]

            new_state[n_index] = self.next_state[index]
        return new_state

    def pivoting(self, index_nonzero, target_size, aux=False):
        """ pivot amplitudes of index_nonzero and self.index_zero"""

        n_qubits = len(self.index_zero)
        target = list(range(n_qubits - target_size))
        remain = list(range(n_qubits - target_size, n_qubits))

        memory = qiskit.QuantumRegister(n_qubits)

        anc, circuit = self._initialize_circuit(memory, remain)

        index_differ = 0

        for k in target:
            if index_nonzero[k] != self.index_zero[k]:
                index_differ = k
                ctrl_state = index_nonzero[k]
                break

        target_cx = []
        for k in target:
            if index_differ != k and index_nonzero[k] != self.index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

        for k in remain:
            if index_nonzero[k] != self.index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

        for k in remain:
            if self.index_zero[k] == '0':
                circuit.x(k)

        if aux:
            # apply mcx using mode v-chain
            mcxvchain(circuit, memory, anc, remain, index_differ)
        else:
            circuit.mcx(remain, index_differ)

        for k in remain:
            if self.index_zero[k] == '0':
                circuit.x(k)

        self.next_state = self._next_state(ctrl_state, index_differ, remain, target_cx)

        return circuit

    def _initialize_circuit(self, memory, remain):
        if self.aux:
            n_anci = len(remain)
            anc = qiskit.QuantumRegister(n_anci - 1, name='anc')
            circuit = qiskit.QuantumCircuit(memory, anc)

        else:
            circuit = qiskit.QuantumCircuit(memory)
            anc = None

        return anc, circuit

    def _get_index_zero(self, non_zero, state):
        index_zero = None
        for k in range(2 ** non_zero):
            txt = '0' + str(self.n_qubits) + 'b'
            index = format(k, txt)

            if index not in state:
                index_zero = index
                break

        return index_zero

    def _get_index_nz(self, target_size):
        index_nonzero = None
        for index, _ in self.next_state.items():
            if index[:target_size] != target_size * '0':
                index_nonzero = index
                break
        return index_nonzero

    @staticmethod
    def initialize(state, aux=False):
        """ Create circuit to initialize a sparse quantum state arXiv:2006.00016

        For instance, to initialize the state a|001>+b|100>
            $ state = {'001': a, '100': b}
            $ circuit = sparse_initialize(state)

        Parameters
        ----------
        state: dict of {str:int}
            A unit vector representing a quantum state.
            Keys are binary strings and values are amplitudes.

        aux: bool
            circuit with auxiliary qubits if aux == True

        Returns
        -------
        sp_circuit: QuantumCircuit
            QuantumCircuit to initialize the state
        """

        return PivotStatePreparation(state, aux).sp_circuit




def mcxvchain(circuit, memory, anc, lst_ctrl, tgt):
    """ multi-controlled x gate with working qubits """
    circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
    for j in range(2, len(lst_ctrl)):
        circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])

    circuit.cx(anc[len(lst_ctrl) - 2], tgt)

    for j in reversed(range(2, len(lst_ctrl))):
        circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])
    circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
