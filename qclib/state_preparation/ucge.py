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
todo
"""
import numpy as np
from qiskit.circuit.library import UCGate
from qclib.state_preparation.ucg import UCGInitialize


def _repetition_verify(base, d, mux, mux_cpy):
    i = 0
    next_base = base + d
    while i < d:
        if not np.allclose(mux[base], mux[next_base]):
            return False
        mux_cpy[next_base] = None
        base, next_base, i = base + 1, next_base + 1, i + 1
    return True


def _repetition_search(mux, n, mux_cpy):

    dont_carry = []
    for i in range(1, len(mux) // 2 + 1):
        entanglement = False
        if np.log2(i).is_integer() and np.allclose(mux[i], mux[0]):
            mux_org = mux_cpy[:]
            repetitions = len(mux) // (2 * i)
            base = 0
            while repetitions:
                repetitions -= 1
                valid = _repetition_verify(base, i, mux, mux_cpy)
                base += 2 * i
                if not valid:
                    mux_cpy[:] = mux_org
                    break
                if repetitions == 0:
                    entanglement = True

        if entanglement:
            dont_carry.append(n + int(np.log2(i)) + 1)
    return dont_carry


class UCGEInitialize(UCGInitialize):
    """ todo """

    def __init__(self, params, label=None, opt_params=None):
        super().__init__(params, label=label, opt_params=opt_params)

    def _define_initialize(self):

        children = self.params
        parent = self._update_parent(children)
        tree_level = self.num_qubits
        r_gate = self.target_state // 2

        while tree_level > 0:

            bit_target, ucg = self._disentangle_qubit(
                children, parent, r_gate, tree_level
            )
            children = self._apply_diagonal(bit_target, parent, ucg)
            parent = self._update_parent(children)

            # prepare next iteration
            r_gate = r_gate // 2
            tree_level -= 1

        self.circuit.global_phase -= sum(np.angle(self.params)) / len(self.params)

        return self.circuit.inverse()

    def _apply_diagonal(
        self,
        bit_target: str,
        parent: "list[float]",
        ucg: UCGate
    ):
        children = parent

        if bit_target == "1":
            diagonal = np.conj(ucg._get_diagonal())[
                1::2
            ]  # pylint: disable=protected-access
        else:
            diagonal = np.conj(ucg._get_diagonal())[
                ::2
            ]  # pylint: disable=protected-access
        if ucg.dont_carry and diagonal.shape[0] > 1:
            # If `diagonal.shape[0] == 1` then diagonal == [1.].
            # Therefore, `diagonal` has no effect on `children`.
            size_required = len(ucg.dont_carry) + len(ucg.controls)
            min_qubit = min([*ucg.dont_carry, *ucg.controls])
            ctrl_qc = [x-min_qubit for x in ucg.controls]

            for i in range(size_required):
                if i not in ctrl_qc:
                    d = 2**i
                    new_diagonal = []
                    n = len(diagonal)
                    for j in range(n):
                        new_diagonal.append(diagonal[j])
                        if (j + 1) % d == 0:
                            new_diagonal.extend(diagonal[j + 1 - d:j + 1])
                    diagonal = np.array(new_diagonal)

        children = children * diagonal

        return children

    def _disentangle_qubit(
        self,
        children: "list[float]",
        parent: "list[float]",
        r_gate: int,
        tree_level: int,
    ):
        """Apply UCGate to disentangle qubit target"""

        bit_target = self.str_target[self.num_qubits - tree_level]

        old_mult, old_controls, target = self._define_mult(children, parent, tree_level)
        nc, mult = self._simplify(old_mult, tree_level)
        mult_controls = [x for x in old_controls if x not in nc]

        if self.preserve:
            self._preserve_previous(mult, mult_controls, r_gate, target)

        ucg = self._apply_ucg(mult, mult_controls, target)
        ucg.dont_carry = nc
        ucg.controls = mult_controls

        return bit_target, ucg

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

    @staticmethod
    def _update_parent(children):

        size = len(children) // 2
        # Calculates norms.
        parent = [
            np.linalg.norm(
                [children[2 * k], children[2 * k + 1]]
            ) for k in range(size)
        ]
        # Calculates phases.
        parent = [
            parent[k] * np.exp(
                1j * np.sum(np.angle([children[2 * k], children[2 * k + 1]])) / 2
            ) for k in range(size)
        ]

        return parent

    def _simplify(self, mux, level):

        mux_cpy = mux.copy()
        dont_carry = []

        if len(mux) > 1:
            n = self.num_qubits - level
            dont_carry = _repetition_search(mux, n, mux_cpy)

        new_mux = [matrix for matrix in mux_cpy if matrix is not None]

        return dont_carry, new_mux

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):

        gate = UCGEInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)
