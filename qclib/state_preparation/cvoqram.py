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

""" cvo-qram """

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import UGate
from qclib.util import _compute_matrix_angles
from qclib.gates.initialize_sparse import InitializeSparse

# pylint: disable=maybe-no-member


class CvoqramInitialize(InitializeSparse):
    """
    Initializing the Amplitude Distribution of a Quantum State

    This class implements a sparse state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        self._name = "cvoqram"
        self._get_num_qubits(params)
        self.norm = 1

        default_with_aux = True
        if opt_params is None:
            self.with_aux = default_with_aux
        else:
            if opt_params.get("with_aux") is None:
                self.with_aux = default_with_aux
            else:
                self.with_aux = opt_params.get("with_aux")

        self._label = label
        if label is None:
            self._label = "SP"

            if inverse:
                self._label = "SPdg"

        super().__init__(self._name, self.num_qubits, params.items(), label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        """Initialize quantum registers"""
        aux = QuantumRegister(1, name="u")
        memory = QuantumRegister(self.num_qubits, name="m")

        if self.with_aux:
            anc = QuantumRegister(self.num_qubits - 1, name="anc")
            circuit = QuantumCircuit(aux, anc, memory)
        else:
            anc = None
            circuit = QuantumCircuit(aux, memory)

        self.norm = 1
        circuit.x(aux[0])
        for k, (binary_string, feature) in enumerate(self.params):
            control = self._select_controls(binary_string)
            self._flip_flop(circuit, control, memory, aux)
            self._load_superposition(circuit, feature, control, memory, anc, aux)
            if k < len(self.params) - 1:
                self._flip_flop(circuit, control, memory, aux)
            else:
                break

        return circuit

    @staticmethod
    def _flip_flop(circuit, control, memory, aux):
        for k in control:
            circuit.cx(aux[0], memory[k])

    @staticmethod
    def _select_controls(binary_string):
        control = []
        for k, bit in enumerate(binary_string[::-1]):
            if bit == "1":
                control.append(k)

        return control

    def _mcuvchain(self, circuit, alpha, beta, phi, control, memory, anc, aux):
        """
        N-qubit controlled-unitary gate
        """

        lst_ctrl = control
        lst_ctrl_reversed = list(reversed(lst_ctrl))
        circuit.rccx(
            memory[lst_ctrl_reversed[0]],
            memory[lst_ctrl_reversed[1]],
            anc[self.num_qubits - 2],
        )

        tof = {}
        i = self.num_qubits - 1
        for ctrl in lst_ctrl_reversed[2:]:
            circuit.rccx(anc[i - 1], memory[ctrl], anc[i - 2])
            tof[ctrl] = [i - 1, i - 2]
            i -= 1

        circuit.cu(alpha, beta, phi, 0, anc[i - 1], aux[0])

        for ctrl in lst_ctrl[:-2]:
            circuit.rccx(anc[tof[ctrl][0]], memory[ctrl], anc[tof[ctrl][1]])

        circuit.rccx(
            memory[lst_ctrl[-1]], memory[lst_ctrl[-2]], anc[self.num_qubits - 2]
        )

    def _load_superposition(self, circuit, feature, control, memory, anc, aux):
        """
        Load pattern in superposition
        """

        theta, phi, lam = _compute_matrix_angles(feature, self.norm)

        if len(control) == 0:
            circuit.u(theta, phi, lam, aux[0])
        elif len(control) == 1:
            circuit.cu(theta, phi, lam, 0, memory[control[0]], aux[0])
        else:
            if self.with_aux:
                self._mcuvchain(circuit, theta, phi, lam, control, memory, anc, aux)
            else:
                gate = UGate(theta, phi, lam).control(len(control))
                circuit.append(gate, memory[control] + [aux[0]])

        self.norm = self.norm - np.absolute(np.power(feature, 2))

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        if qubits is None:
            q_circuit.append(
                CvoqramInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(CvoqramInitialize(state, opt_params=opt_params), qubits)
