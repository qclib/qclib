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
from qiskit.quantum_info import Operator
from qclib.util import _compute_matrix_angles
from qclib.gates.initialize_sparse import InitializeSparse
from qclib.gates.mcg import mcg


# pylint: disable=maybe-no-member


QuantumCircuit.mcg = mcg


class CvoqramInitialize(InitializeSparse):
    """
    Initializing the Amplitude Distribution of a Quantum State

    This class implements a sparse state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
        self._get_num_qubits(params)
        self.norm = 1
        self.anc = None
        self.aux = None
        self.memory = None
        self.control = None

        default_with_aux = True
        if opt_params is None:
            self.with_aux = default_with_aux
            self.use_linear_mcg = True
        else:
            if opt_params.get("with_aux") is None:
                self.with_aux = default_with_aux
            else:
                self.with_aux = opt_params.get("with_aux")

            if opt_params.get("use_linear_mcg") is None:
                self.use_linear_mcg = True
            else:
                self.use_linear_mcg = opt_params.get("use_linear_mcg")

        if label is None:
            label = "CVOSP"

        super().__init__("cvoqram", self.num_qubits, params.items(), label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        """Initialize quantum registers"""

        self.anc = QuantumRegister(self.num_qubits - 1, name="anc")
        self.aux = QuantumRegister(1, name="u")
        self.memory = QuantumRegister(self.num_qubits, name="m")
        if self.with_aux:
            circuit = QuantumCircuit(self.aux, self.anc, self.memory)
        else:
            self.anc = None
            circuit = QuantumCircuit(self.aux, self.memory)

        self.norm = 1
        circuit.x(self.aux[0])
        for k, (binary_string, feature) in enumerate(self.params):
            self.control = self._select_controls(binary_string)
            self._flip_flop(circuit)
            self._load_superposition(circuit, feature, self.control, self.memory)
            if k < len(self.params) - 1:
                self._flip_flop(circuit)
            else:
                break

        return circuit

    def _flip_flop(self, circuit):
        for k in self.control:
            circuit.cx(self.aux[0], self.memory[k])

    @staticmethod
    def _select_controls(binary_string):
        control = []
        for k, bit in enumerate(binary_string[::-1]):
            if bit == "1":
                control.append(k)

        return control

    def _mcuvchain(self, circuit, alpha, beta, phi):
        """
        N-qubit controlled-unitary gate
        """

        lst_ctrl = self.control
        lst_ctrl_reversed = list(reversed(lst_ctrl))
        circuit.rccx(
            self.memory[lst_ctrl_reversed[0]],
            self.memory[lst_ctrl_reversed[1]],
            self.anc[self.num_qubits - 2],
        )

        tof = {}
        i = self.num_qubits - 1
        for ctrl in lst_ctrl_reversed[2:]:
            circuit.rccx(self.anc[i - 1], self.memory[ctrl], self.anc[i - 2])
            tof[ctrl] = [i - 1, i - 2]
            i -= 1

        circuit.cu(alpha, beta, phi, 0, self.anc[i - 1], self.aux[0])

        for ctrl in lst_ctrl[:-2]:
            circuit.rccx(self.anc[tof[ctrl][0]], self.memory[ctrl], self.anc[tof[ctrl][1]])

        circuit.rccx(
            self.memory[lst_ctrl[-1]], self.memory[lst_ctrl[-2]], self.anc[self.num_qubits - 2]
        )

    def _load_superposition(self, circuit, feature, control, memory):
        """
        Load pattern in superposition
        """

        theta, phi, lam = _compute_matrix_angles(feature, self.norm)

        if len(control) == 0:
            circuit.u(theta, phi, lam, self.aux[0])
        elif len(control) == 1:
            circuit.cu(theta, phi, lam, 0, memory[control[0]], self.aux[0])
        else:
            if self.with_aux:
                self._mcuvchain(circuit, theta, phi, lam)
            else:
                gate = UGate(theta, phi, lam)
                if self.use_linear_mcg:
                    gate_op = Operator(gate).data
                    circuit.mcg(gate_op, memory[control], [self.aux[0]])
                else:
                    circuit.append(gate.control(len(control)), memory[control] + [self.aux[0]])

        self.norm = self.norm - np.absolute(np.power(feature, 2))

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        if qubits is None:
            q_circuit.append(
                CvoqramInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(CvoqramInitialize(state, opt_params=opt_params), qubits)
