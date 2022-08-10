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
https://arxiv.org/abs/1501.06911
https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.state_preparation.initialize import Initialize
from qclib.isometry import decompose


class IsometryInitialize(Initialize):
    """
    Isometry state preparation
    https://arxiv.org/abs/1501.06911
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.93.032318

    This class implements an isometry based state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.

        opt_params: {'scheme': scheme}
            scheme: str
                method to decompose the isometry ('knill', 'ccd', 'csd', 'qiskit').
                Default is scheme='ccd'.
        """
        self._name = "isometry"
        self._get_num_qubits(params)

        if opt_params is None:
            self.scheme = "ccd"
        else:
            if opt_params.get("scheme") is None:
                self.scheme = "ccd"
            else:
                self.scheme = opt_params.get("scheme")

        self._label = label
        if label is None:
            self._label = "SP"

            if inverse:
                self._label = "SPdg"

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        if self.scheme == "qiskit":
            reg = QuantumRegister(self.num_qubits)
            circuit = QuantumCircuit(reg)
            # pylint: disable=maybe-no-member
            circuit.isometry(
                np.array(self.params), q_input=[], q_ancillas_for_output=reg
            )
            return circuit

        return decompose(np.array(self.params), scheme=self.scheme)

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a IsometryInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                IsometryInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(IsometryInitialize(state, opt_params=opt_params), qubits)
