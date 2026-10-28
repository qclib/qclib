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
State preparation with SynQ (https://github.com/qsoftware/synq), a C++
synthesis library. The circuit uses uniformly controlled RY and RZ rotations
(Mottonen et al., https://arxiv.org/abs/quant-ph/0407010).

SynQ is an optional dependency; see ``qclib.synq``.
"""

import numpy as np
from qclib.gates.initialize import Initialize
from qclib import synq


class SynqInitialize(Initialize):
    """
    State preparation gate synthesized by SynQ.
    """

    def __init__(self, params, label=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.
        """
        self._name = "synq"
        self._get_num_qubits(params)

        if label is None:
            label = "SynqSP"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = synq.state_preparation(np.array(self.params, dtype=complex))

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        """
        Appends a SynqInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(SynqInitialize(state), q_circuit.qubits)
        else:
            q_circuit.append(SynqInitialize(state), qubits)
