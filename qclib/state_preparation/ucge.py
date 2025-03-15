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


def _first_and_second_halves_equal(base: int, d: int, mux: "list[np.ndarray]"):
    """
    Returns True if mux[base : base + d] = mux[next_base : next_base + d]
    """

    next_base = base + d
    return np.allclose(mux[base : base + d], mux[next_base : next_base + d])


def _repetition_search(mux: "list[np.ndarray]", reversed_level: int):
    """
    Search for possible partitions by searching for equal operators in mux[0] and mux[d],
    where d is power of two. When a possible partition is found, it calculates the position
    of the controls and operators to be eliminated

    Parameters
    ----------
    mux: List of 2 x 2 unitary gates representing a multiplexer
    reversed_level

    Returns
    -------
    deleted_controls: controls that must be removed from the multiplexer
    deleted_operators: index of operators that must be removed from the multiplexer
    """

    deleted_operators = set()
    deleted_controls = []

    for d in [2 ** int(j) for j in range(0, int(np.log2(len(mux))))]:
        delete_set = set()
        if np.allclose(mux[d], mux[0]):
            delete_set = _find_operators_to_remove(d, mux)

        if delete_set:
            removed_control = reversed_level + int(np.log2(d)) + 1
            deleted_controls.append(removed_control)
            deleted_operators.update(delete_set)

    return deleted_controls, deleted_operators


def _find_operators_to_remove(d, mux):
    """
    Verifies if mux can be split into len(mux) // (2 * d) sequential partitions, where
    the operators in the first half of each partition is equal to the second half. If
    the two halves of all partitions are equal, the indexes of the second halves of
    the partitions are returned to be later removed from the multiplexer.

    Parameters
    ----------
    d: a power of two integer in range(0, len(mux)
    mux: List of 2 x 2 unitary gates representing a multiplexer

    Returns
    -------
    deleted_operators: index of operators that must be removed from the multiplexer

    """

    deleted_operators = set()
    num_partitions = len(mux) // (2 * d)
    base = 0

    for _ in range(num_partitions, 0, -1):
        if _first_and_second_halves_equal(base, d, mux):
            deleted_operators.update(range(base + d, base + 2 * d))
            base += 2 * d
        else:
            deleted_operators = set()
            break

    return deleted_operators


class UCGEInitialize(UCGInitialize):
    """
    This class implements an efficient state preparation for separable states
    Based on the UCG approach

    https://arxiv.org/abs/2409.05618
    """

    def __init__(self, params, label=None, opt_params=None):
        super().__init__(params, label=label, opt_params=opt_params)

    # pylint: disable=arguments-differ
    def _apply_diagonal(self, bit_target: str, parent: "list[float]", ucg: UCGate):
        children = parent
        # pylint: disable=protected-access
        if bit_target == "1":
            diagonal = np.conj(ucg._get_diagonal())[1::2]
        else:
            diagonal = np.conj(ucg._get_diagonal())[::2]
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
        nc, mult = self._simplify(old_mult)
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

                # avoid a global phase difference in the operators
                temp = children[2 * k] / value
                if temp.real < 0:
                    new_parent.append(-value)
                else:
                    new_parent.append(value)

            parent = new_parent

        return parent

    def _simplify(self, mux: "list[np.ndarray]"):
        """
        Remove redundant gates and operators from the multiplexer

        Parameters
        ----------
        mux: List of 2 x 2 unitary gates representing a multiplexer
        level: level of the multiplexer in the state preparation tree

        Returns
        -------
        removed_controls: controls that must be removed of the multiplexer
        simplified_mux: multiplexer without the redundant gates
        """

        deleted_operators = set()
        removed_controls = []

        if len(mux) > 1:
            level = np.log2(len(mux)) + 1
            reversed_level = self.num_qubits - level

            removed_controls, deleted_operators = _repetition_search(mux, reversed_level)

        if deleted_operators:
            simplified_mux = [mux[k] for k in range(len(mux)) if k not in deleted_operators]
            return removed_controls, simplified_mux

        simplified_mux = mux
        return removed_controls, simplified_mux

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):

        gate = UCGEInitialize(state, opt_params=opt_params)
        if qubits is None:
            q_circuit.append(gate.definition, q_circuit.qubits)
        else:
            q_circuit.append(gate.definition, qubits)
