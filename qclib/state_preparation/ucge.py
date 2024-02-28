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

"""
import numpy as np
import qiskit
from qiskit.circuit.library import UCGate
from qiskit.quantum_info import Operator
from qclib.state_preparation.ucg import UCGInitialize


def _repetition_verify(a, b, d, mux, mux_cpy):
    i = 0
    while i < d:
        if not np.allclose(mux[a], mux[b]):
            return False
        mux_cpy[b] = np.array([[-999, -999], [-999, -999]])
        a, b, i = a + 1, b + 1, i + 1
    return True


def _repetition_search(mux, n, mux_cpy):

    nc = []
    for i in range(1, len(mux)):
        if i > len(mux) / 2:
            return nc
        d = i
        entanglement = False
        if np.allclose(mux[i], mux[0]) and (d & (d - 1)) == 0:
            mux_org = mux_cpy[:]
            repetitions = len(mux) / (2 * d)
            base = 0
            while repetitions:
                repetitions -= 1
                valid = _repetition_verify(base, base + d, d, mux, mux_cpy)
                base += 2 * d
                if not valid:
                    mux_cpy[:] = mux_org
                    break
                if repetitions == 0:
                    entanglement = True

        if entanglement:
            nc.append(n + int(np.log2(d)) + 1)
    return nc


class UCGEInitialize(UCGInitialize):
    """

    """

    def __init__(self, params, label=None, opt_params=None):
        super().__init__(params, label=label, opt_params=opt_params)

    def _define_initialize(self):

        children = self.params
        parent = self._update_parent(children)
        tree_level = self.num_qubits
        r_gate = self.target_state // 2

        while tree_level > 0:

            bit_target, ucg, nc, controls = self._disentangle_qubit(children, parent, r_gate, tree_level)
            children = self._apply_diagonal(self, bit_target, parent, ucg, nc, controls)
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

        old_mult, old_controls, target = self._define_mult(children, parent, tree_level)
        nc, mult = self._simplify(old_mult, tree_level)
        mult_controls = [x for x in old_controls if x not in nc]

        if self.preserve:
            self._preserve_previous(mult, mult_controls, r_gate, target)

        ucg = self._apply_ucg(mult, mult_controls, target)

        return bit_target, ucg, nc, mult_controls

    def _simplify(self, mux, level):

        mux_cpy = mux.copy()
        nc = []

        if len(mux) > 1:
            n = self.num_qubits - level
            nc = _repetition_search(mux, n, mux_cpy)

        new_mux = [i for i in mux_cpy if not np.allclose(i, np.array([[-999, -999], [-999, -999]]))]

        return nc, new_mux

    @staticmethod
    def _apply_diagonal(self, bit_target: str, parent: 'list[float]', ucg: UCGate, nc: 'list[int]', controls: 'list[int]'):
        children = parent

        if bit_target == '1':
            diagonal = np.conj(ucg._get_diagonal())[1::2]  # pylint: disable=protected-access
        else:
            diagonal = np.conj(ucg._get_diagonal())[::2]  # pylint: disable=protected-access
        if nc:
            controls.reverse()
            size_required = len((nc + controls))
            ctrl_qc = [self.num_qubits - 1 - x for x in controls]
            unitary_diagonal = np.diag(diagonal)
            qc = qiskit.QuantumCircuit(size_required)
            qc.unitary(unitary_diagonal, ctrl_qc)
            matrix = Operator(qc).to_matrix()
            diagonal = np.diag(matrix)
        children = children * diagonal

        return children

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):

        gate = UCGEInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)
