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
n-qubit controlled gate
"""
from collections import namedtuple
import numpy as np
import qiskit
from qiskit.circuit import Gate
from qiskit import QuantumCircuit, QuantumRegister
from ._utils import _check_u2, _apply_ctrl_state

# pylint: disable=maybe-no-member
# pylint: disable=protected-access
class McGate(Gate):

    def __init__(self, unitary, num_controls, ctrl_state: str = None):

        _check_u2(unitary)

        self.unitary = unitary

        if num_controls > 0:
            self.control_qubits = QuantumRegister(num_controls)
        else:
            self.control_qubits = []

        self.target_qubit = QuantumRegister(1)

        self.num_qubits = num_controls + 1

        self.ctrl_state = ctrl_state

        super().__init__("mc_gate", self.num_qubits, [], "mc_gate")

    def _define(self):

        if len(self.control_qubits) > 0:
            self.definition = QuantumCircuit(self.control_qubits, self.target_qubit)

            self._apply_ctrl_state()

            qubits_indexes = list(range(self.num_qubits))

            gate_circuit = qiskit.QuantumCircuit(self.num_qubits, name="T" + str(0))

            self._c1c2(self.unitary, self.num_qubits, gate_circuit)
            self._c1c2(self.unitary, self.num_qubits, gate_circuit, step=-1)

            self._c1c2(self.unitary, self.num_qubits - 1, gate_circuit, False)
            self._c1c2(self.unitary, self.num_qubits - 1, gate_circuit, False, step=-1)

            self.definition.append(gate_circuit, [*self.control_qubits, self.target_qubit])

            self._apply_ctrl_state()

        else:
            self.definition = QuantumCircuit(self.target_qubit)
            self.definition.unitary(self.unitary, 0)

    def _c1c2(self, unitary, n_qubits, gate_circ, first=True, step=1):
        pairs = namedtuple("pairs", ["control", "target"])

        if step == 1:
            start = 0
            reverse = True
        else:
            start = 1
            reverse = False

        qubit_pairs = [
            pairs(control, target)
            for target in range(n_qubits)
            for control in range(start, target)
        ]

        qubit_pairs.sort(key=lambda e: e.control + e.target, reverse=reverse)

        for pair in qubit_pairs:
            exponent = pair.target - pair.control
            if pair.control == 0:
                exponent = exponent - 1
            param = 2**exponent
            signal = -1 if (pair.control == 0 and not first) else 1
            signal = step * signal
            if pair.target == n_qubits - 1 and first:
                csqgate = McGate._gate_u(unitary, param, signal)
                gate_circ.compose(csqgate,
                                  qubits=[pair.control, pair.target],
                                  inplace=True)
            else:
                gate_circ.crx(signal * np.pi / param, pair.control, pair.target)

    @staticmethod
    def _gate_u(agate, coef, signal):
        param = 1 / np.abs(coef)

        values, vectors = np.linalg.eig(agate)
        gate = np.power(values[0] + 0j, param) * vectors[:, [0]] @ vectors[:, [0]].conj().T
        gate = (
            gate
            + np.power(values[1] + 0j, param) * vectors[:, [1]] @ vectors[:, [1]].conj().T
        )

        if signal < 0:
            gate = np.linalg.inv(gate)

        sqgate = QuantumCircuit(1, name="U^1/" + str(coef))
        sqgate.unitary(gate, 0)  # pylint: disable=maybe-no-member
        csqgate = sqgate.control(1)

        return csqgate

    @staticmethod
    def mc_gate(circuit, unitary, controls, target, ctrl_state=None):
        circuit.append(
            McGate(unitary, len(controls), ctrl_state=ctrl_state),
            [*controls, target]
        )

McGate._apply_ctrl_state = _apply_ctrl_state
