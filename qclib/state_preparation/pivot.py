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
from qclib.state_preparation import LowRankInitialize
from qclib.state_preparation.initialize_sparse import InitializeSparse

# pylint: disable=maybe-no-member


class PivotInitialize(InitializeSparse):
    """ Pivot State Preparation """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        self._name = 'pivot'
        self._get_num_qubits(params)

        default_aux = False
        if opt_params is None:
            self.aux = default_aux
        else:
            if opt_params.get('aux') is None:
                self.aux = default_aux
            else:
                self.aux = opt_params.get('aux')

        self._label = label
        if label is None:
            self._label = 'SP'

            if inverse:
                self._label = 'SPdg'

        super().__init__(self._name, self.num_qubits, params.items(), label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        non_zero = len(self.params)
        target_size = np.log2(non_zero)
        target_size = np.ceil(target_size)
        target_size = int(target_size)

        memory = qiskit.QuantumRegister(self.num_qubits, name='q')
        if self.aux:
            remain = list(range(self.num_qubits - target_size, self.num_qubits))
            n_anci = len(remain)

            anc = qiskit.QuantumRegister(n_anci - 1, name='anc')
            pivot_circuit = qiskit.QuantumCircuit(anc, memory)

        else:
            pivot_circuit = qiskit.QuantumCircuit(memory)

        next_state = self.params.copy()
        index_nonzero = self._get_index_nz(self.num_qubits - target_size, next_state)
        while index_nonzero is not None:
            index_zero = self._get_index_zero(non_zero, next_state)
            circ, next_state = self._pivoting(index_nonzero, target_size, index_zero, next_state)
            pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)
            index_nonzero = self._get_index_nz(self.num_qubits - target_size, next_state)

        dense_state = np.zeros(2 ** target_size, dtype=complex)
        for key, value in next_state:
            dense_state[int(key, 2)] = value

        if non_zero <= 2:
            initialize_circ = qiskit.QuantumCircuit(1)
            initialize_circ.initialize(dense_state)
        else:
            initialize_circ = QuantumCircuit(target_size)
            LowRankInitialize.initialize(initialize_circ, dense_state)

        if self.aux:
            circuit = qiskit.QuantumCircuit(anc, memory)
            nun_aux = n_anci - 1
            circuit.compose(
                initialize_circ,
                circuit.qubits[nun_aux:nun_aux + target_size],
                inplace=True)

            circuit.barrier()
            circuit.compose(pivot_circuit.reverse_bits().reverse_ops(), inplace=True)
        else:
            circuit = qiskit.QuantumCircuit(self.num_qubits)
            circuit.compose(
                initialize_circ,
                circuit.qubits[:target_size], inplace=True)

            circuit.compose(
                pivot_circuit.reverse_bits().reverse_ops(),
                circuit.qubits,
                inplace=True)

        return circuit

    @staticmethod
    def _next_state(ctrl_state, index_differ, remain, target_cx, index_zero, next_state):
        tab = {'0': '1', '1': '0'}
        new_state = {}
        for index, amp in next_state:
            if index[index_differ] == ctrl_state:
                n_index = ''
                for k, _ in enumerate(index):
                    if k in target_cx:
                        n_index = n_index + tab[index[k]]
                    else:
                        n_index = n_index + index[k]
            else:
                n_index = index

            if n_index[remain[0]:] == index_zero[remain[0]:]:
                n_index = n_index[:index_differ] + \
                          tab[index[index_differ]] + n_index[index_differ + 1:]

            new_state[n_index] = amp

        return new_state.items()

    def _pivoting(self, index_nonzero, target_size, index_zero, next_state):
        """ pivot amplitudes of index_nonzero and self.index_zero"""

        target = list(range(self.num_qubits - target_size))
        remain = list(range(self.num_qubits - target_size, self.num_qubits))

        memory = qiskit.QuantumRegister(self.num_qubits)

        anc, circuit = self._initialize_circuit(memory, remain)

        index_differ = 0

        for k in target:
            if index_nonzero[k] != index_zero[k]:
                index_differ = k
                ctrl_state = index_nonzero[k]
                break

        target_cx = []
        for k in target:
            if index_differ != k and index_nonzero[k] != index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

        for k in remain:
            if index_nonzero[k] != index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

        for k in remain:
            if index_zero[k] == '0':
                circuit.x(k)

        if self.aux:
            # apply mcx using mode v-chain
            self._mcxvchain(circuit, memory, anc, remain, index_differ)
        else:
            circuit.mcx(remain, index_differ)

        for k in remain:
            if index_zero[k] == '0':
                circuit.x(k)

        next_state = self._next_state(ctrl_state, index_differ, remain,
                                      target_cx, index_zero, next_state)

        return circuit, next_state

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
            index = f'{k:0{self.num_qubits}b}'

            not_exists = sum([1 for v in state if v[0] == index]) == 0
            if not_exists:
                index_zero = index
                break

        return index_zero

    @staticmethod
    def _get_index_nz(target_size, next_state):
        index_nonzero = None
        for index, _ in next_state:
            if index[:target_size] != target_size * '0':
                index_nonzero = index
                break
        return index_nonzero

    @staticmethod
    def _mcxvchain(circuit, memory, anc, lst_ctrl, tgt):
        """ multi-controlled x gate with working qubits """
        circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
        for j in range(2, len(lst_ctrl)):
            circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])

        circuit.cx(anc[len(lst_ctrl) - 2], tgt)

        for j in reversed(range(2, len(lst_ctrl))):
            circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])
        circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        if qubits is None:
            q_circuit.append(PivotInitialize(state, opt_params=opt_params), q_circuit.qubits)
        else:
            q_circuit.append(PivotInitialize(state, opt_params=opt_params), qubits)
