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
Bidirectional state preparation
https://arxiv.org/abs/2108.10182
"""

from math import ceil
import numpy as np
from qiskit import QuantumCircuit

from qclib.gates.initialize_adaptive import InitializeAdaptive
from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    sparse_state_decomposition
)
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util.tree_register import add_register
from qclib.state_preparation.util.tree_walk import top_down, bottom_up
from qclib.state_preparation.util.tree_utils import leftmost, is_leaf

class AdaptiveInitialize(InitializeAdaptive):
    """
    This class implements a state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.

        opt_params: {'split': split}
            split: int
                Level (enumerated from bottom to top, where 1 ≤ s ≤ n)
                at which the angle tree is split.
                Default value is ``ceil(n/2)`` (sublinear).
        """

        bit_string = next(iter(params))
        tree_height = len(bit_string)
        if opt_params is None:
            self.split = int(ceil(tree_height / 2))  # sublinear
            self.reset_ancillae = False
            self.global_phase = False
        else:
            self.split = opt_params.get("split", int(ceil(tree_height / 2)))
            if self.split is None:
                self.split = int(ceil(tree_height / 2))
            self.reset_ancillae = opt_params.get("reset_ancillae", False)
            self.global_phase = opt_params.get("global_phase", False)

        self._dict_params = {i: params[i] for i in sorted(params)}

        self._name = "adaptative"
        self.n_output = tree_height
        self._get_num_qubits(self._dict_params)

        if label is None:
            label = "ADSP"

        super().__init__(
            self._name,
            self.num_qubits,
            [v for _, v in self._dict_params.items()],
            label=label
        )

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        bit_string = next(iter(self._dict_params))
        n_qubits = len(bit_string)
        data = [Amplitude(int(i, base=2), a) for i, a in self._dict_params.items()]

        state_tree = sparse_state_decomposition(n_qubits, data)
        angle_tree = create_angles_tree(state_tree)

        circuit = QuantumCircuit()
        start_level = n_qubits - self.split

        AdaptiveInitialize._calculate_substates(state_tree, start_level)
        AdaptiveInitialize._entaglement_analysis(angle_tree, start_level)

        add_register(circuit, angle_tree, start_level)

        top_down(angle_tree, circuit, start_level)
        bottom_up(angle_tree, circuit, start_level)

        if self.global_phase:
            params = [v for _, v in self._dict_params.items()]
            circuit.global_phase += sum(np.angle(params)) / 2**n_qubits

        if self.reset_ancillae:
            circuit.reset(range(n_qubits, circuit.num_qubits))

        return circuit

    @staticmethod
    def _calculate_substates(state_tree, start_level):
        if state_tree:
            AdaptiveInitialize._calculate_substates(state_tree.left, start_level)
            AdaptiveInitialize._calculate_substates(state_tree.right, start_level)

            if is_leaf(state_tree):
                substate = np.array([complex(1.0)])
                qubit_state = np.array([complex(1.0)])
            else:
                beta = state_tree.beta
                lmbda = state_tree.lmbda
                left_state = 0
                right_state = 0

                if state_tree.left:
                    left_state = np.sqrt(1.0-beta**2)*np.exp(-1j*lmbda) * np.kron([1, 0], state_tree.left.substate)
                if state_tree.right:
                    right_state = beta*np.exp(+1j*lmbda) * np.kron([0, 1], state_tree.right.substate)

                substate = left_state + right_state

                qubit_state = (
                    np.sqrt(1.0-beta**2)*np.exp(-1j*lmbda) * np.array([1, 0]) +
                    beta*np.exp(+1j*lmbda) * np.array([0, 1])
                )

            state_tree.substate = substate
            state_tree.qubit_state = qubit_state

    @staticmethod
    def _entaglement_analysis(angle_tree, start_level):
        if angle_tree:
            if angle_tree.left and angle_tree.right:
                if np.allclose(
                    angle_tree.left.state_node.substate,
                    angle_tree.right.state_node.substate
                ):
                    angle_tree.right = None
                    angle_tree.state_node.right = None

            AdaptiveInitialize._entaglement_analysis(angle_tree.left, start_level)
            AdaptiveInitialize._entaglement_analysis(angle_tree.right, start_level)

            left = angle_tree.left
            right = angle_tree.right

            while left and right and (right.level < start_level or (right.level == start_level and is_leaf(right) and is_leaf(left))):
                if np.allclose(
                        left.state_node.qubit_state,
                        right.state_node.qubit_state
                ):
                    right.pruned = True
                else:
                    angle_tree.pruned = False
                    left.pruned = False
                    right.pruned = False

                    control_state = angle_tree.state_node
                    left_state = left.state_node
                    right_state = right.state_node

                    left_state.qubit_state = (
                        np.sqrt(1.0-control_state.beta**2)*np.exp(-1j*control_state.lmbda) * left_state.qubit_state+
                        control_state.beta*np.exp(+1j*control_state.lmbda) * right_state.qubit_state
                    )
                    right_state.qubit_state = (
                        np.sqrt(1.0-control_state.beta**2)*np.exp(-1j*control_state.lmbda) * right_state.qubit_state+
                        control_state.beta*np.exp(+1j*control_state.lmbda) * left_state.qubit_state
                    )
                    left_state.qubit_state = left_state.qubit_state / np.linalg.norm(left_state.qubit_state)
                    right_state.qubit_state = right_state.qubit_state / np.linalg.norm(right_state.qubit_state)

                left = leftmost(left)
                right = leftmost(right)

    def _get_num_qubits(self, params):
        bit_string = next(iter(params))
        tree_height = len(bit_string)
        self.num_qubits = (self.split + 1) * 2 ** (tree_height - self.split) - 1

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a AdaptiveInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                AdaptiveInitialize(
                    state,
                    opt_params=opt_params),
                q_circuit.qubits
            )
        else:
            q_circuit.append(
                AdaptiveInitialize(
                    state,
                    opt_params=opt_params),
                qubits
            )
