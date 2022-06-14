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
Divide-and-conquer state preparation
https://doi.org/10.1038/s41598-021-85474-1
https://arxiv.org/abs/2108.10182
"""

from math import log2
from qiskit import QuantumCircuit

from qclib.state_preparation.initialize import Initialize
from qclib.state_preparation.util.state_tree_preparation import Amplitude, state_decomposition
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util.tree_register import add_register
from qclib.state_preparation.util.tree_walk import bottom_up

class DcspInitialize(Initialize):
    """
    A divide-and-conquer algorithm for quantum state preparation
    https://doi.org/10.1038/s41598-021-85474-1

    This class implements a state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None):
        """
            Parameters
            ----------
            params: list of complex
                A unit vector representing a quantum state.
                Values are amplitudes.

        """
        self._name = 'dcsp'
        self._get_num_qubits(params)

        self._label = label
        if label is None:
            self._label = 'SP'

            if inverse:
                self._label = 'SPdg'

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        n_qubits = int(log2(len(self.params)))
        data = [Amplitude(i, a) for i, a in enumerate(self.params)]

        state_tree = state_decomposition(n_qubits, data)
        angle_tree = create_angles_tree(state_tree)

        circuit = QuantumCircuit()
        add_register(circuit, angle_tree, n_qubits-1)

        bottom_up(angle_tree, circuit, n_qubits)

        return circuit

    def _get_num_qubits(self, params):
        if not log2(len(params)).is_integer():
            Exception("The number of amplitudes is not a power of 2")
        self.num_qubits = len(params)-1

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        """
        Appends a DcspInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(DcspInitialize(state), q_circuit.qubits)
        else:
            q_circuit.append(DcspInitialize(state), qubits)
