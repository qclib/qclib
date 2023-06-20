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
Initializes a mixed quantum state.
"""

from math import log2, ceil, isclose
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.gates.initialize import Initialize
from qclib.gates.initialize_mixed import InitializeMixed
from qclib.gates.initialize_sparse import InitializeSparse
from qclib.state_preparation import LowRankInitialize

# pylint: disable=maybe-no-member


class MixedInitialize(InitializeMixed):
    """
    This class implements a mixed state preparation gate.
    """

    def __init__(
            self,
            params,
            initializer=LowRankInitialize,
            opt_params=None,
            probabilities=None,
            label=None,
            reset=True,
            classical = True
    ):
        """
        Parameters
        ----------
        params: list of list of complex
            A list of unit vectors, each representing a quantum state.
            Values are amplitudes.

        initializer: Initialize or InitializeSparse
            Type of the class that will be applied to prepare pure states.
            Default is ``LowRankInitialize``.

        opt_params: dictionary
            Optional parameters of the class of type ``initializer``.

        reset: bool
            Indicates whether the auxiliary qubits should be reset or not.
            Default is ``True``.

        classical: bool
            Indicates whether the purification is done classically or
            in-circuit (quantically). Default is ``True``.
        """

        if (
            not issubclass(initializer, Initialize) and
            not issubclass(initializer, InitializeSparse)
        ):
            raise TypeError("The value of initializer should be Initialize or InitializeSparse.")

        if probabilities is None:
            probabilities = [1/len(params)] * len(params)
        elif any(i < 0.0 for i in probabilities):
            raise ValueError("All probabilities must greater than or equal to 0.")
        elif any(i > 1.0 for i in probabilities):
            raise ValueError("All probabilities must less than or equal to 1.")
        elif not isclose(sum(probabilities), 1.0):
            raise ValueError("The sum of the probabilities must be 1.0.")

        self._name = "mixed"
        self._get_num_qubits(params)

        self._initializer = initializer
        self._reset = reset
        self._opt_params = opt_params
        self._probabilities = probabilities
        self._classical = classical

        self._num_ctrl_qubits = int(ceil(log2(len(params))))
        self._num_data_qubits = initializer(params[0]).num_qubits

        self._list_params = params

        if label is None:
            label = "Mixed"

        super().__init__(self._name, self.num_qubits, np.array(params).reshape(-1), label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        purified_circuit = QuantumCircuit(self.num_qubits)

        if self._classical:
            # Calculates the pure state classically.
            pure_state = np.zeros(2**(self._num_qubits), dtype=complex)
            for index, (state_vector, prob) in enumerate(
                zip(self._list_params, self._probabilities)
            ):
                basis = np.zeros(2**self._num_ctrl_qubits)
                basis[index] = 1

                pure_state += np.kron(np.sqrt(prob) * state_vector, basis)

            purified_circuit = self._initializer(
                pure_state,
                opt_params=self._opt_params
            ).definition

        else:
            # Calculates the pure state quantically.
            aux_state = np.concatenate((
                np.sqrt(self._probabilities),
                [0] * (2**(self._num_ctrl_qubits) - len(self._probabilities))
            ))

            sub_circuit = self._initializer(
                aux_state,
                opt_params=self._opt_params
            ).definition

            sub_circuit.name = 'aux. space'

            purified_circuit.append(sub_circuit, range(self._num_ctrl_qubits))

            for index, state in enumerate(self._list_params):
                sub_circuit = self._initializer(
                    state,
                    opt_params=self._opt_params
                ).definition

                sub_circuit.name = f'state {index}'

                sub_circuit = sub_circuit.control(
                    num_ctrl_qubits=self._num_ctrl_qubits,
                    ctrl_state = f"{index:0{self._num_ctrl_qubits}b}"
                )

                purified_circuit.compose(sub_circuit, purified_circuit.qubits, inplace=True)

        purified_circuit.name = 'purified state'

        circuit = QuantumCircuit()
        circuit.add_register(QuantumRegister(self._num_ctrl_qubits, 'aux'))
        circuit.add_register(QuantumRegister(self._num_data_qubits, 'rho'))
        circuit.append(purified_circuit, circuit.qubits)

        if self._reset:
            circuit.reset(range(self._num_ctrl_qubits))

        return circuit

    @staticmethod
    def initialize(q_circuit, ensemble, qubits=None, opt_params=None, probabilities=None):
        """
        Appends a MixedInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                MixedInitialize(
                    ensemble,
                    opt_params=opt_params,
                    probabilities=probabilities
                )
                , q_circuit.qubits
            )
        else:
            q_circuit.append(
                MixedInitialize(
                    ensemble,
                    opt_params=opt_params,
                    probabilities=probabilities
                )
                , qubits
            )
