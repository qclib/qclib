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
        https://arxiv.org/abs/quant-ph/0407010
        https://arxiv.org/abs/2108.10182
"""

import numpy as np
from qiskit import QuantumCircuit

from qclib.state_preparation.initialize import Initialize
from qclib.state_preparation.util.state_tree_preparation import Amplitude, state_decomposition
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util.tree_register import add_register
from qclib.state_preparation.util.tree_walk import top_down


class TopDownInitialize(Initialize):
    """
    Top-down state preparation
    https://arxiv.org/abs/quant-ph/0407010
    https://arxiv.org/abs/2108.10182

    This class implements a state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        """
            Parameters
            ----------
            params: list of complex
                A unit vector representing a quantum state.
                Values are amplitudes.

            opt_params: {'global_phase': global_phase}
                global_phase: bool
                    If ``True``, corrects the global phase.
                    Default value is ``True``.
        """
        self._name = 'top-down'
        self._get_num_qubits(params)

        if opt_params is None:
            self.global_phase = True
        else:
            if opt_params.get('global_phase') is None:
                self.global_phase = True
            else:
                self.global_phase = opt_params.get('global_phase')

        self._label = label
        if label is None:
            self._label = 'SP'

            if inverse:
                self._label = 'SPdg'

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        data = [Amplitude(i, a) for i, a in enumerate(self.params)]

        state_tree = state_decomposition(self.num_qubits, data)
        angle_tree = create_angles_tree(state_tree)

        circuit = QuantumCircuit()
        add_register(circuit, angle_tree, 0)

        top_down(angle_tree, circuit, 0)
        if self.global_phase:
            circuit.global_phase += sum(np.angle(self.params))/len(self.params)

        return circuit

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a TopDownInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(TopDownInitialize(state, opt_params=opt_params), q_circuit.qubits)
        else:
            q_circuit.append(TopDownInitialize(state, opt_params=opt_params), qubits)
