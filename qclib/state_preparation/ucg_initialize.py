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
from qclib.gates.initialize import Initialize
from qclib.gates.uc_gate import UCGate


class UCGInitialize(Initialize):
    """
        Quantum circuits with uniformly controlled one-qubit gates
        https://doi.org/10.48550/arXiv.quant-ph/0410066
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):

        self._name = "ucg_initialize"
        self._get_num_qubits(params)
        self.target_state = 0 if opt_params is None else opt_params.get("target_state")

        if label is None:
            label = "ucg_initialize"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        string_target = bin(self.target_state)[2:].zfill(self.num_qubits)[::-1]
        q_register = QuantumRegister(self.num_qubits)
        q_circuit = QuantumCircuit(q_register)

        children = self.params
        size = len(children) // 2
        parent = [la.norm([children[2 * k], children[2 * k + 1]]) for k in range(size)]

        tree_level = self.num_qubits
        while tree_level > 0:
            bit_target = string_target[self.num_qubits-tree_level]
            current_level_mux = self._build_multiplexor(parent,
                                                        children,
                                                        target=bit_target)
            ucg = UCGate(current_level_mux, up_to_diagonal=True)

            controls = q_register[self.num_qubits - tree_level + 1:]
            target = [q_register[self.num_qubits - tree_level]]
            q_circuit.append(ucg, target + controls)

            # preparing for the next loop
            tree_level -= 1
            children = parent
            if bit_target == '1':
                diagonal = np.conj(ucg._get_diagonal())[1::2]  # pylint: disable=protected-access
            else:
                diagonal = np.conj(ucg._get_diagonal())[::2]  # pylint: disable=protected-access

            children = children * diagonal
            size = len(children) // 2
            parent = [la.norm([children[2 * k], children[2 * k + 1]]) for k in range(size)]

        return q_circuit.inverse()

    def _build_multiplexor(self, parent_amplitudes, children_amplitudes, target='0'):
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
        gates = []
        len_pnodes = len(parent_amplitudes)
        len_snodes = len(children_amplitudes)
        for parent_idx, sibling_idx in zip(range(len_pnodes), range(0, len_snodes, 2)):
            if parent_amplitudes[parent_idx] != 0:
                amp_ket0 = (children_amplitudes[sibling_idx] / parent_amplitudes[parent_idx])
                amp_ket1 = (children_amplitudes[sibling_idx + 1] / parent_amplitudes[parent_idx])
                gates += [self._get_branch_operator(amp_ket0, amp_ket1, target)]
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
