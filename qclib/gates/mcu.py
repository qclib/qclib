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
from qclib.gates.util import check_u2, apply_ctrl_state
from qclib.gates.multitargetmcsu2 import MultiTargetMCSU2
from qclib.gates.ldmcu import Ldmcu


# pylint: disable=protected-access
class MCU(Gate):
    """
    Approximated Multi-Controlled Unitary Gate
    https://arxiv.org/abs/2310.14974

    -----------------------------------------
    Implements gate decomposition of a multi-controlled operator in U(2)
    """

    def __init__(self, unitary, num_controls, error=0, ctrl_state: str = None):
        """

        Parameters
        ----------
        unitary: U(2) gate
        num_controls (int): Number of controls
        error (float): max_error
        ctrl_state (str or int): Control state in decimal or as a bitstring
        """
        check_u2(unitary)

        self.unitary = unitary
        self.error = error

        if num_controls > 0:
            self.control_qubits = QuantumRegister(num_controls)
        else:
            self.control_qubits = []

        self.target_qubit = QuantumRegister(1)

        self.n_ctrl_base = self._get_num_base_ctrl_qubits(self.unitary, self.error)
        if self.n_ctrl_base == 0:
            raise ValueError("The number of base qubits is 0")
        if self.n_ctrl_base > num_controls:
            raise ValueError("The number of control qubits is too low")

        self.ctrl_state = ctrl_state

        super().__init__("McuApprox", num_controls + 1, [], "McuApprox")

    def _define(self):
        if len(self.control_qubits) > 0:
            self.definition = QuantumCircuit(self.control_qubits, self.target_qubit)

            self._apply_ctrl_state()

            gate_circuit = qiskit.QuantumCircuit(self.num_qubits, name="T" + str(0))

            self._c1c2(self.num_qubits, gate_circuit)
            self._c1c2(self.num_qubits, gate_circuit, step=-1)

            self._c1c2(self.num_qubits - 1, gate_circuit, False)
            self._c1c2(self.num_qubits - 1, gate_circuit, False, -1)

            self.definition.append(
                gate_circuit, [*self.control_qubits, self.target_qubit]
            )

            self._apply_ctrl_state()

        else:
            self.definition = QuantumCircuit(self.target_qubit)
            self.definition.unitary(self.unitary, 0)

    @staticmethod
    def _get_num_base_ctrl_qubits(unitary, error):
        """
        Get the baseline number of control qubits for the approximation
        given an error
        args:
        unitary: Unitary to be approximated
        error: Error of the approximation
        """
        eig_vals, _ = np.linalg.eig(unitary)
        angles = np.angle(eig_vals)

        if (1 - np.cos(angles[0])) >= (1 - np.cos(angles[1])):
            angle = angles[0]

        else:
            angle = angles[1]

        quotient = angle / np.arccos(1 - error**2 / 2)
        return int(np.ceil(np.log2(quotient))) + 1

    def get_n_base(self, unitary, error):
        return self._get_num_base_ctrl_qubits(unitary, error)

    def _c1c2(self, n_qubits, gate_circ, first=True, step=1):

        extra_q, n_qubits_base = self._calc_extra_qubits(first, n_qubits)

        qubit_pairs = self._compute_qubit_pairs(n_qubits_base, step)

        unitary_list = []
        targets = []

        for pair in qubit_pairs:
            param = self._compute_param(pair)
            signal = -1 if (pair.control == 0 and not first) else 1
            signal = step * signal
            # Perform a translation of the qubits by extra_q qubits in all cases
            # When target == last qubit and first==true apply the U gates, except
            # when control==0, in which case we don't do anything
            if pair.target == n_qubits_base - 1 and first:
                if pair.control != 0:
                    csqgate = MCU._gate_u(self.unitary, param, signal)
                    gate_circ.compose(
                        csqgate,
                        qubits=[pair.control + extra_q, pair.target + extra_q],
                        inplace=True,
                    )
            # For the controlled rotations, when control==0, apply a multi-controlled
            # rotation with the extra control qubits
            else:
                if pair.control == 0 and extra_q >= 1:
                    # Apply a multi-controlled Rx gate with the additional control qubits
                    control_list = np.array(range(0, extra_q + 1))

                    unitary_list.append(self._compute_rx_matrix(param, signal))
                    targets.append(pair.target + extra_q)

                    if pair.target == 1:
                        MultiTargetMCSU2.multi_target_mcsu2(
                            gate_circ, unitary_list, control_list, targets
                        )
                else:
                    gate_circ.crx(
                        signal * np.pi / param,
                        pair.control + extra_q,
                        pair.target + extra_q,
                    )

    def _calc_extra_qubits(self, first, n_qubits):
        if first:
            n_qubits_base = self.n_ctrl_base + 1
        else:
            n_qubits_base = self.n_ctrl_base
        extra_q = n_qubits - n_qubits_base
        return extra_q, n_qubits_base

    @staticmethod
    def _compute_param(pair):
        exponent = pair.target - pair.control
        if pair.control == 0:
            exponent = exponent - 1
        param = 2**exponent
        return param

    @staticmethod
    def _compute_rx_matrix(param, signal):
        theta = signal * np.pi / param
        rx_matrix = np.array(
            [
                [np.cos(theta / 2), (-1j) * np.sin(theta / 2)],
                [(-1j) * np.sin(theta / 2), np.cos(theta / 2)],
            ]
        )
        return rx_matrix

    @staticmethod
    def _compute_qubit_pairs(n_qubits_base, step):
        pairs = namedtuple("pairs", ["control", "target"])
        if step == 1:
            start = 0
            reverse = True
        else:
            start = 1
            reverse = False
        qubit_pairs = [
            pairs(control, target)
            for target in range(n_qubits_base)
            for control in range(start, target)
        ]
        qubit_pairs.sort(key=lambda e: e.control + e.target, reverse=reverse)
        return qubit_pairs

    @staticmethod
    def _gate_u(a_gate, coefficient, signal):
        param = 1 / np.abs(coefficient)

        values, vectors = np.linalg.eig(a_gate)
        gate = (
            np.power(values[0] + 0j, param) * vectors[:, [0]] @ vectors[:, [0]].conj().T
        )
        gate = (
            gate
            + np.power(values[1] + 0j, param)
            * vectors[:, [1]]
            @ vectors[:, [1]].conj().T
        )

        if signal < 0:
            gate = np.linalg.inv(gate)

        s_q_gate = QuantumCircuit(1, name="U^1/" + str(coefficient))
        s_q_gate.unitary(gate, 0)  # pylint: disable=maybe-no-member
        c_s_q_gate = s_q_gate.control(1)

        return c_s_q_gate

    @staticmethod
    def mcu(circuit, unitary, controls, target, error, ctrl_state=None):
        """
        Approximated Multi-Controlled Unitary Gate
        https://arxiv.org/abs/2310.14974
        """
        if error == 0:
            circuit.append(
                Ldmcu(unitary, len(controls), ctrl_state=ctrl_state),
                [*controls, target]
            )
        else:
            circuit.append(
                MCU(unitary, len(controls), error, ctrl_state=ctrl_state),
                [*controls, target],
            )


MCU._apply_ctrl_state = apply_ctrl_state
