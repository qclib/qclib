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
Implements the state preparation
defined at https://link.springer.com/article/10.1007/s11128-010-0177-y
"""

from math import log2, pi

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qclib.gates.initialize import Initialize
from qclib.gates.ucr import ucr

# pylint: disable=maybe-no-member

class FrqiInitialize(Initialize):
    """
    A flexible representation of quantum images for polynomial
    preparation, image compression, and processing operations
    https://link.springer.com/article/10.1007/s11128-010-0177-y

    This class implements a state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of angles
            A vector representing an image.
            Values are angles representing color.

        opt_params: {'rescale': rescale
                     'method': method}
            rescale: bool
                If `True`, it rescales the values of the `params`
                vector to the range between 0 and pi.
            method: method
                Scheme used to decompose uniformed controlled rotations.
                Possible values are ``'ucr'`` (multiplexer) and ``'mcr'``
                (multicontrolled rotations).
                Default is ``method='ucr'``.
        """
        self._name = "frqi"

        if opt_params is None:
            self.rescale = False
            self.method = 'ucr'
        else:
            self.rescale = False if opt_params.get("rescale") is None else opt_params.get("rescale")
            self.method = 'ucr' if opt_params.get("method") is None else opt_params.get("method")

        scaled_params = params
        if self.rescale:
            scaled_params = (
                (np.array(params) - np.min(self.params)) /
                (np.max(params) - np.min(params)) * pi/2
            )

        self._get_num_qubits(scaled_params)

        if label is None:
            label = "FRQI"

        super().__init__(self._name, self.num_qubits, scaled_params, label=label)

    def _get_num_qubits(self, params):
        self.num_qubits = log2(len(params))

        # Check if param is a power of 2
        if self.num_qubits == 0 or not self.num_qubits.is_integer():
            raise ValueError("The length of the state vector is not a positive power of 2.")

        # Check if any pixels values is not between 0 and pi/2
        if any(0 > x > pi/2 for x in params):
            raise ValueError("All pixel values must be between 0 and pi/2.")

        self.num_qubits = int(self.num_qubits) + 1

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        circuit = QuantumCircuit(self.num_qubits)
        circuit.h(circuit.qubits[:-1])

        if self.method == 'ucr':
            circuit.compose(
                ucr(RYGate, self.params),
                circuit.qubits[:-1],
                inplace=True
            )
        else:
            pass

        return circuit

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a FrqiInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                FrqiInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(FrqiInitialize(state, opt_params=opt_params), qubits)
