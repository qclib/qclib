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
      This module implements the state preparation proposed by
      Bergholm et al (2005) available in:
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330
"""

import math
import numpy as np
import numpy.linalg as la
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions.quantum_initializer.uc import UCGate
from qclib.gates.initialize import Initialize

class UCGInitialize(Initialize):
    """
        Quantum circuits with uniformly controlled one-qubit gates
        https://doi.org/10.48550/arXiv.quant-ph/0410066
    """

    def __init__(self, params, label=None, opt_params=None):

        self._name = "ucg_initialize"
        self._get_num_qubits(params)
        self.register = QuantumRegister(self.num_qubits)
        self.circuit = QuantumCircuit(self.register)
        self.target_state = 0 if opt_params is None else opt_params.get("target_state")
        self.str_target = bin(self.target_state)[2:].zfill(self.num_qubits)[::-1]
        self.preserve = False if opt_params is None else opt_params.get("preserve_previous")

        if label is None:
            label = "ucg_initialize"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        children = self.params
        parent = self._update_parent(children)
        tree_level = self.num_qubits
        r_gate = self.target_state // 2

        while tree_level > 0:

            bit_target, ucg = self._disentangle_qubit(children, parent, r_gate, tree_level)
            # children = self._apply_diagonal(bit_target, parent, ucg)
            children = parent
            parent = self._update_parent(children)

            # prepare next iteration
            r_gate = r_gate // 2
            tree_level -= 1

        return self.circuit.inverse()

    def _disentangle_qubit(self, children: 'list[float]',
                           parent: 'list[float]',
                           r_gate: int, tree_level: int):
        """ Apply UCGate to disentangle qubit target"""
        bit_target = self.str_target[self.num_qubits - tree_level]

        mults = []
        controls = []
        mult, mult_controls, target = self._define_mult(children, parent, tree_level)
        mults.append(mult)
        controls.append(mult_controls)

        if self.preserve:
            self._preserve_previous(mult, mult_controls, r_gate, target)

        mult_simp = self._simplify(mult)

        isSeparable = True

        for i in mult_simp:
            if len(i) == math.log2(len(mult)):
                isSeparable = False
                break

        if isSeparable:
            mults.clear()
            controls.clear()
            for i in mult_simp:
                mults.append(mult_simp[i])
                controls.append(list(i))

        ucg = self._apply_ucg(mults, controls, target)

        return bit_target, ucg

    def _define_mult(self, children: 'list[float]', parent: 'list[float]', tree_level: int):

        current_level_mux = self._build_multiplexor(parent,
                                                    children,
                                                    self.str_target)
        mult_controls, target = self._get_ctrl_targ(tree_level)

        return current_level_mux, mult_controls, target

    def _apply_ucg(self, current_level_mux: 'list[list[np.ndarray]]',
                   mult_controls: 'list[list[int]]',
                   target: int):
        """ Creates and applies multiplexer """


        for i, mux in enumerate(current_level_mux):
            ucg = UCGate(current_level_mux[i], up_to_diagonal=False)
            if len(mux) != 1:
                self.circuit.append(ucg, [target] + mult_controls[i])
            else:
                self.circuit.unitary(mux[0], target) # pylint: disable=maybe-no-member

        return ucg

    def _simplify(self, table):
        """ simplifies the multiplexer """

        simple = {}
        ops = set()
        for array in table:
            ops.add(tuple(array.flatten()))
        ops = [np.array(arr).reshape(array.shape) for arr in ops]

        size = len(table)

        for i in ops:
            v = []
            for j in range(size):
                if np.allclose(i, table[j]):
                    key = format(j, f"0{int(math.log2(size))}b")
                    v.append(key)
            tp = []
            while tp != v:
                tp = v.copy()
                self._reduction(v)
            index = tuple(i.flatten())
            simple[index] = v

        dict_ops_pos = {}
        dict_ops_ctrl = {}

        set_controls = set()
        ctrls_empty = 0
        for i in simple:
            pos = []
            ctrl = []
            for j in simple[i]:
                controls = set()
                index = 0
                for k in j:
                    if k != '_' and size > 1:
                        controls.add(self.num_qubits - index - 1)
                    index += 1
                t_ctrl = tuple(controls)
                set_controls.add(t_ctrl)
                string_pos = j.replace("_", "")
                ctrl.append(controls)
                if string_pos != "":
                    pos.append(int(string_pos, 2))
                else:
                    ctrls_empty = 1

            dict_ops_pos[tuple(i)] = pos
            dict_ops_ctrl[tuple(i)] = ctrl

        dict_mult = {}
        for i in set_controls:
            vec = []
            for j in range(pow(2, len(i))):
                vec.append(np.identity(2))
            dict_mult[i] = vec

        for i in dict_ops_pos:
            if ctrls_empty:
                op = np.array(i)
                op_form = op.reshape((2, 2))
                empty_tp = ()
                vec = [op_form]
                dict_mult[empty_tp] = vec
            else:
                t = 0
                for j in enumerate(dict_ops_pos[i]):
                    t_ctrl = tuple(dict_ops_ctrl[i][t])
                    position = dict_ops_pos[i][t]
                    op = np.array(i)
                    op_form = op.reshape((2, 2))
                    dict_mult[t_ctrl][position] = op_form
                    t += 1
        return dict_mult

    def _reduction(self, v):
        """ apply the reduction to the multiplexer"""

        applied_simp = self._better_simp(v)
        for it in applied_simp.indexes:
            v.remove(it)
        if applied_simp.simp:
            v.append(applied_simp.simp)
        return v

    def _better_simp(self, v):
        """ finds the best simplification that can be applied """

        class Item:
            def __init__(self, indexes, simp, size):
                self.indexes = indexes
                self.simp = simp
                self.simps = set()
                self.size = size

        new_v = v[:]
        w = []
        for key1 in new_v:
            for key2 in new_v:
                if self._hamming(key1, key2) == 1:
                    set_index = {key1, key2}
                    item = Item(set_index, self._filter_controls(key1, key2), len(set_index))
                    item.simps.add(item.simp)
                    w.append(item)
        temp = self._better_simp_aux(w)
        while temp != w:
            w = temp
            temp = self._better_simp_aux(temp)

        if len(w):

            max_simp = {}
            max_value = 0

            for value in w:
                if value.size > max_value:
                    max_value = value.size
            for value in w:
                if value.size == max_value:
                    max_simp[value.simp] = [value, 0]

            for index in max_simp:
                factor = 0
                pair = max_simp[index]
                indexes = v[:]
                for ind in pair[0].indexes:
                    indexes.remove(ind)
                for i in indexes:
                    for j in indexes:
                        if self._hamming(i, j) == 1:
                            factor += 1
                max_simp[index][1] = factor

            max_factor = 0
            for e in max_simp:
                if max_factor == 0:
                    max_index = e
                if max_simp[e][1] > max_factor:
                    max_factor = max_simp[e][1]
                    max_index = e
            btt = max_simp[max_index][0]

        else:
            btt = Item(set(), "", 0)

        return btt

    def _better_simp_aux(self, v):
        """ auxiliary function to better_simp """

        class Item:
            def __init__(self, indexes, simp, size):
                self.indexes = indexes
                self.simp = simp
                self.simps = set()
                self.size = size

        new_list = v[:]
        for it1 in new_list:
            for it2 in new_list:
                simp1 = it1.simp
                simp2 = it2.simp
                if self._hamming(simp1, simp2) == 1:
                    find = any(simp1 in obj.simps and simp2 in obj.simps for obj in new_list)
                    if not find:
                        set_index = it1.indexes | it2.indexes
                        item = Item(set_index, self._filter_controls(simp1, simp2), len(set_index))
                        item.simps.update([simp1, simp2])
                        if not any(item.indexes <= obj.indexes for obj in new_list):
                            new_list.append(item)
        return new_list

    def _hamming(self, string1, string2):
        """ Calculates the hamming distance between two strings """

        distance = sum(c1 != c2 for c1, c2 in zip(string1, string2))
        return distance

    def _filter_controls(self, string1, string2):
        """ Creates a new element by eliminating the different control """

        return ''.join('_' if c1 != c2 else c1 for c1, c2 in zip(string1, string2))

    def _preserve_previous(self, mux: 'list[np.ndarray]',
                           mult_controls: 'list[int]',
                           r_gate: int, target: int):
        """
        Remove one gate from mux and apply separately to avoid changing previous base vectors
        """
        out_gate = mux[r_gate]
        qc_gate = QuantumCircuit(1)
        qc_gate.unitary(out_gate, 0)  # pylint: disable=maybe-no-member
        mux[r_gate] = np.eye(2)

        out_gate_ctrl = list(range(0, target)) + list(range(target + 1, self.num_qubits))
        ctrl_state = self.str_target[0:target][::-1]

        if len(ctrl_state) < self.num_qubits - 1:
            ctrl_state = bin(r_gate)[2:].zfill(len(mult_controls)) + ctrl_state

        gate = qc_gate.control(self.num_qubits - 1, ctrl_state=ctrl_state)

        self.circuit.compose(gate, out_gate_ctrl + [target], inplace=True)

    @staticmethod
    def _update_parent(children):

        size = len(children) // 2
        parent = [la.norm([children[2 * k], children[2 * k + 1]]) for k in range(size)]

        return parent

    @staticmethod
    def _apply_diagonal(bit_target: str, parent: 'list[float]', ucg: UCGate):

        children = parent
        if bit_target == '1':
            diagonal = np.conj(ucg._get_diagonal())[1::2]  # pylint: disable=protected-access
        else:
            diagonal = np.conj(ucg._get_diagonal())[::2]  # pylint: disable=protected-access
        children = children * diagonal

        return children

    def _get_ctrl_targ(self, tree_level: int):

        controls = list(range(self.num_qubits - tree_level + 1, self.num_qubits))
        target = self.num_qubits - tree_level

        return controls, target

    def _build_multiplexor(self, parent_amplitudes: 'list[float]',
                           children_amplitudes: 'list[float]', str_target: str):
        """
        Infers the unitary to be used in the uniformily controlled multiplexor
        defined by Bergholm et al (2005).
        Args:
        parent_amplitudes: list of amplitudes
        children_amplitudes: children of the parent amplitudes
        Returns:
        list of 2-by-2 numpy arrays with the desired unitaries to be used
        in the multiplexor
        """

        tree_lvl = int(np.log2(len(children_amplitudes)))
        bit_target = str_target[self.num_qubits - tree_lvl]
        gates = []

        len_pnodes = len(parent_amplitudes)
        len_snodes = len(children_amplitudes)

        for parent_idx, sibling_idx in zip(range(len_pnodes), range(0, len_snodes, 2)):
            if parent_amplitudes[parent_idx] != 0:
                amp_ket0 = (children_amplitudes[sibling_idx] / parent_amplitudes[parent_idx])
                amp_ket1 = (children_amplitudes[sibling_idx + 1] / parent_amplitudes[parent_idx])
                if amp_ket0 != 0:
                    gates += [self._get_branch_operator(amp_ket0, amp_ket1, bit_target)]
                else:
                    gates += [self._get_diagonal_operator(amp_ket1, bit_target)]
            else:
                gates += [np.eye(2)]
        return gates

    @staticmethod
    def _get_branch_operator(amplitude_ket0, amplitude_ket1, target='0'):
        """
        Returns the matrix operator that is going to split the qubit in to two components
        of a superposition
        Args:
        amplitude_ket0: Complex amplitude of the |0> component of the superpoisition
        amplitude_ket1: Complex amplitude of the |1> component of the superpoisition
        Returns:
        A 2x2 numpy array defining the desired operator inferred from
        the amplitude argument
        """

        if target == '0':
            operator = np.array([[amplitude_ket0, -np.conj(amplitude_ket1)],
                                 [amplitude_ket1, np.conj(amplitude_ket0)]])
        else:
            operator = np.array([[-np.conj(amplitude_ket1), amplitude_ket0],
                                 [np.conj(amplitude_ket0), amplitude_ket1]])

        return np.conj(operator).T

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):

        gate = UCGInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)

    @staticmethod
    def _get_diagonal_operator(amplitude_ket1, target):

        if target == '0':
            operator = np.array([[0, 1], [amplitude_ket1, 0]])
        else:
            operator = np.array([[1, 0], [0, amplitude_ket1]])

        return np.conj(operator).T
