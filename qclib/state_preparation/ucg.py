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
import numpy as np
import numpy.linalg as la
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions.quantum_initializer import UCGate
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
            children = self._apply_diagonal(bit_target, parent, ucg)
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

        mult, mult_controls, target = self._define_mult(children, parent, tree_level)

        if self.preserve:
            self._preserve_previous(mult, mult_controls, r_gate, target)

        ucg = self._apply_ucg(mult, mult_controls, target)

        return bit_target, ucg

    def _define_mult(self, children: 'list[float]', parent: 'list[float]', tree_level: int):

        current_level_mux = self._build_multiplexor(parent,
                                                    children,
                                                    self.str_target)
        mult_controls, target = self._get_ctrl_targ(tree_level)

        return current_level_mux, mult_controls, target

    def _apply_ucg(self, current_level_mux: 'list[np.ndarray]',
                   mult_controls: 'list[int]',
                   target: int):
        """ Creates and applies multiplexer """

        ucg = UCGate(current_level_mux, up_to_diagonal=True)
        if len(current_level_mux) != 1:
            self.circuit.append(ucg, [target] + mult_controls)
        else:
            self.circuit.unitary(current_level_mux[0], target) # pylint: disable=maybe-no-member
        return ucg

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
                amp_ket0 = children_amplitudes[sibling_idx] / parent_amplitudes[parent_idx]
                amp_ket1 = children_amplitudes[sibling_idx + 1] / parent_amplitudes[parent_idx]
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
