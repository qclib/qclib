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

from math import ceil, log2
import numpy as np
from qiskit import QuantumCircuit

from qclib.gates.initialize import Initialize
from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    state_decomposition,
)
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util.tree_register import add_register
from qclib.state_preparation.util.tree_walk import top_down, bottom_up


class BdspInitialize(Initialize):
    """
    Configurable sublinear circuits for quantum state preparation
    https://arxiv.org/abs/2108.10182

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
        if opt_params is None:
            self.split = int(ceil(log2(len(params)) / 2))  # sublinear
            self.global_phase = False
        else:
            self.global_phase = opt_params.get("global_phase", False)
            if opt_params.get("split") is None:
                self.split = int(ceil(log2(len(params)) / 2))  # sublinear
            else:
                self.split = opt_params.get("split")

        self._name = "bdsp"
        self._get_num_qubits(params)

        if label is None:
            label = "BDSP"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        n_qubits = int(np.log2(len(self.params)))
        data = [Amplitude(i, a) for i, a in enumerate(self.params)]

        state_tree = state_decomposition(n_qubits, data)
        angle_tree = create_angles_tree(state_tree)

        circuit = QuantumCircuit()
        add_register(circuit, angle_tree, n_qubits - self.split)

        top_down(angle_tree, circuit, n_qubits - self.split)
        bottom_up(angle_tree, circuit, n_qubits - self.split)

        if self.global_phase:
            circuit.global_phase += sum(np.angle(self.params)) / 2**n_qubits

        return circuit

    def _get_num_qubits(self, params):
        n_qubits = log2(len(params))
        if not n_qubits.is_integer():
            raise ValueError("The number of amplitudes is not a power of 2")
        n_qubits = int(n_qubits)
        self.num_qubits = (self.split + 1) * 2 ** (n_qubits - self.split) - 1

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a BdspInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                BdspInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(BdspInitialize(state, opt_params=opt_params), qubits)
