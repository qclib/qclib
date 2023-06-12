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
TODO
"""

from math import log2, ceil
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.gates.initialize import Initialize
from qclib.gates.initialize_mixed import InitializeMixed
from qclib.gates.initialize_sparse import InitializeSparse
from qclib.state_preparation import LowRankInitialize

# pylint: disable=maybe-no-member


class MixedInitialize(InitializeMixed):
    """
    TODO
    """

    def __init__(
            self,
            params,
            initializer=LowRankInitialize,
            opt_params=None,
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

        opt_params: TODO
        """

        if (
            not issubclass(initializer, Initialize) and
            not issubclass(initializer, InitializeSparse)
        ):
            print(initializer)
            raise TypeError("The value of initializer should be Initialize or InitializeSparse.")

        self._name = "mixed"
        self._get_num_qubits(params)

        self._initializer = initializer
        self._reset = reset
        self._opt_params = opt_params
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
            for index, state_vector in enumerate(self._list_params):
                basis = np.zeros(2**self._num_ctrl_qubits)
                basis[index] = 1

                pure_state += np.kron(state_vector, basis)

            # Completes the index register.
            zero_state = np.zeros(2**self._num_data_qubits)
            zero_state[0] = 1
            for index in range(len(self._list_params), 2**self._num_ctrl_qubits):
                basis = np.zeros(2**self._num_ctrl_qubits)
                basis[index] = 1

                pure_state += np.kron(zero_state, basis)

            # Normalizes the pure_state.
            pure_state = (1/np.sqrt(2**self._num_ctrl_qubits)) * pure_state

            purified_circuit = self._initializer(
                pure_state,
                opt_params=self._opt_params
            ).definition

        else:
            purified_circuit.h(range(self._num_ctrl_qubits))

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

                purified_circuit.append(sub_circuit, purified_circuit.qubits)

        purified_circuit.name = 'purified state'

        circuit = QuantumCircuit()
        circuit.add_register(QuantumRegister(self._num_ctrl_qubits, 'aux'))
        circuit.add_register(QuantumRegister(self._num_data_qubits, 'data'))
        circuit.append(purified_circuit, circuit.qubits)

        if self._reset:
            circuit.reset(range(self._num_ctrl_qubits))

        return circuit

    @staticmethod
    def initialize(q_circuit, states, qubits=None, opt_params=None):
        """
        Appends a MixedInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                MixedInitialize(states, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(MixedInitialize(states, opt_params=opt_params), qubits)
