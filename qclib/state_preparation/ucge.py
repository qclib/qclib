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
Efficient version of the UCG approach for separable states
https://arxiv.org/abs/2409.05618
"""
import numpy as np
from qiskit.circuit.library import UCGate
from qclib.state_preparation.ucg import UCGInitialize
from typing import Set


def _first_and_second_halves_equal(
    base: int, d: int, mux: "list[np.ndarray]"
):
    """
    Checks whether a possible repeating pattern is valid by checking whether all elements repeat
    in a period d and marks operators to be removed
    """

    next_base = base + d

    if not np.allclose(mux[base:base+d], mux[next_base:next_base+d]):
        return False

    return True


def _repetition_search(mux: "list[np.ndarray]", n: int):
    """
    Search for possible repetitions by searching for equal operators in indices that are
    powers of two When found, it calculates the position of the controls to be eliminated
    """
    deleted_operators = set()
    dont_carry = []
    for i in [2 ** int(j) for j in range(0, int(np.log2(len(mux))))]:
        not_entangled = False
        delete_set = set()
        if np.allclose(mux[i], mux[0]):
            not_entangled, delete_set = is_dont_care(i, mux)

        if not_entangled:
            dont_carry.append(n + int(np.log2(i)) + 1)
            deleted_operators.update(delete_set)
    return dont_carry, deleted_operators


def is_dont_care(d, mux):

    deleted_operators = set()
    not_entangled = True
    repetitions = len(mux) // (2 * d)
    base = 0
    for _ in range(repetitions, 0, -1):
        if _first_and_second_halves_equal(base, d, mux):
            deleted_operators.update(range(base + d, base + 2 * d))
            base += 2 * d
        else:
            not_entangled = False
            break

    return not_entangled, deleted_operators


class UCGEInitialize(UCGInitialize):
    """
    This class implements an efficient state preparation for separable states
    Based on the UCG approach

    https://arxiv.org/abs/2409.05618
    """

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

        return self.circuit.inverse()

    # pylint: disable=arguments-differ
    def _apply_diagonal(self, bit_target: str, parent: "list[float]", ucg: UCGate):
        children = parent

        if bit_target == "1":
            diagonal = np.conj(ucg._get_diagonal())[1::2]  # pylint: disable=protected-access
        else:
            diagonal = np.conj(ucg._get_diagonal())[::2]  # pylint: disable=protected-access
        if ucg.dont_carry and diagonal.shape[0] > 1:
            # If `diagonal.shape[0] == 1` then diagonal == [1.].
            # Therefore, `diagonal` has no effect on `children`.
            size_required = len(ucg.dont_carry) + len(ucg.controls)
            min_qubit = min([*ucg.dont_carry, *ucg.controls])
            ctrl_qc = [x - min_qubit for x in ucg.controls]

            # Adjusts the diagonal to the right size
            # Necessary when a simplification occurs
            for i in range(size_required):
                if i not in ctrl_qc:
                    d = 2**i
                    new_diagonal = []
                    n = len(diagonal)

                    # Extends the operator to the total number of qubits in the circuit
                    # This acts as identity on non-target qubits
                    for j in range(n):
                        new_diagonal.append(diagonal[j])
                        if (j + 1) % d == 0:
                            new_diagonal.extend(diagonal[j + 1 - d : j + 1])

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

    @staticmethod
    def _update_parent(children):

        size = len(children) // 2
        # Calculates norms.
        parent = [
            np.linalg.norm([children[2 * k], children[2 * k + 1]]) for k in range(size)
        ]

        # Calculates phases.
        new_parent = []

        if size > 1:
            for k in range(size):
                angle = np.angle([children[2 * k], children[2 * k + 1]])
                angle = angle % (2 * np.pi)
                phase = np.sum(angle) / 2
                value = parent[k] * np.exp(1j * phase)

                temp = children[2 * k] / value
                if temp.real < 0:
                    new_parent.append(-value)
                else:
                    new_parent.append(value)


            parent = new_parent

        return parent

    def _simplify(self, mux: "list[np.ndarray]", level: int):
        """
        Returns the position of controls that can be eliminated and the simplified multiplexer
        """

        deleted_operators = set()
        dont_carry = []
        new_mux = mux
        if len(mux) > 1:
            n = self.num_qubits - level
            dont_carry, deleted_operators = _repetition_search(mux, n)
        if deleted_operators:
            new_mux = mux.copy()
            for k in deleted_operators:
                try:
                    new_mux[k] = None
                except IndexError:
                    pass
            new_mux = [matrix for matrix in new_mux if matrix is not None]

        return dont_carry, new_mux

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):

        gate = UCGEInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)
