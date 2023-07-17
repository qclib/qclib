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
import math
import numpy as np
import qiskit
from qiskit.circuit import Gate
from qiskit import QuantumCircuit, QuantumRegister
from qclib.gates.util import check_u2, apply_ctrl_state
from qclib.gates.cldmcsu import Cldmcsu


# pylint: disable=maybe-no-member
# pylint: disable=protected-access
class LdmcuApprox(Gate):
    """
    Linear Depth Multi-Controlled Unitary
    -----------------------------------------
    Ajeitar descrição (WIP)
    Implements gate decomposition of a munticontrolled operator in U(2) according to
    https://arxiv.org/abs/2203.11882
    https://journals.aps.org/pra/abstract/10.1103/PhysRevA.106.042602.
    """

    def __init__(self, unitary, num_controls, error, ctrl_state: str = None):

        check_u2(unitary)

        self.unitary = unitary
        self.error = error

        if num_controls > 0:
            self.control_qubits = QuantumRegister(num_controls)
        else:
            self.control_qubits = []

        self.target_qubit = QuantumRegister(1)

        self.num_qubits = num_controls + 1

        self.n_ctrl_base = self._get_num_base_ctrl_qubits(self.unitary, self.error)
        if self.n_ctrl_base == 0:
            raise Exception("The number of base qubits is 0")
        if self.n_ctrl_base > num_controls:
            raise Exception("The number of control qubits is too low")

        self.ctrl_state = ctrl_state

        super().__init__("LdmcuApprox", self.num_qubits, [], "LdmcuApprox")

    def _define(self):

        if len(self.control_qubits) > 0:
            self.definition = QuantumCircuit(self.control_qubits, self.target_qubit)

            self._apply_ctrl_state()

            gate_circuit = qiskit.QuantumCircuit(self.num_qubits, name="T" + str(0))

            self._c1c2(self.unitary, self.num_qubits, self.n_ctrl_base, gate_circuit)
            self._c1c2(self.unitary, self.num_qubits, self.n_ctrl_base, gate_circuit, step=-1)
            # Attention to the number of qubits
            self._c1c2(self.unitary, self.num_qubits - 1, self.n_ctrl_base, gate_circuit, False)
            self._c1c2(self.unitary, self.num_qubits - 1, self.n_ctrl_base, gate_circuit,
                       False, step=-1)

            self.definition.append(gate_circuit, [*self.control_qubits, self.target_qubit])

            self._apply_ctrl_state()

        else:
            self.definition = QuantumCircuit(self.target_qubit)
            self.definition.unitary(self.unitary, 0)

    def _get_num_base_ctrl_qubits(self, unitary, error):
        '''
        Get the baseline number of control qubits for the approximation
        given an error
        args:
        unitary: Unitary to be approximated
        error: Error of the approximation
        '''
        eig_vals, eig_vecs = np.linalg.eig(unitary)
        angles = np.angle(eig_vals)

        if (1 - np.cos(angles[0])) >= (1 - np.cos(angles[1])):
                angle = angles[0]

        else:
                angle = angles[1]

        quotient = angle / np.arccos(1 - error**2/2)
        return int(np.ceil(np.log2(quotient)))


    def _c1c2(self, unitary, n_qubits, n_ctrl_base, gate_circ, first=True, step=1):

        pairs = namedtuple("pairs", ["control", "target"])
        # Get the number of extra qubits
        if first:
            n_qubits_base = n_ctrl_base + 1
        else:
            n_qubits_base = n_ctrl_base
        extra_q = n_qubits - n_qubits_base

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

        unitary_list = []
        targets = []

        for pair in qubit_pairs:
            exponent = pair.target - pair.control
            if pair.control == 0:
                exponent = exponent - 1
            param = 2 ** exponent
            signal = -1 if (pair.control == 0 and not first) else 1
            signal = step * signal
            # Perform a translation of the qubits by extra_q qubits in all cases
            # When target == last qubit and first==true apply the U gates, except
            # when control==0, in which case we don't do anything
            if pair.target == n_qubits_base - 1 and first:
                if pair.control != 0:
                    csqgate = LdmcuApprox._gate_u(unitary, param, signal)
                    gate_circ.compose(csqgate,
                                      qubits=[pair.control + extra_q, pair.target + extra_q],
                                      inplace=True)
            # For the controlled rotations, when control==0, apply a multicontrolled
            # rotation with the extra control qubits
            else:
                if pair.control == 0 and extra_q >= 1:
                    # Apply a multi-controlled Rx gate with the additional control qubits
                    control_list = np.array(range(0, extra_q + 1))
                    theta = signal * np.pi / param
                    rx_matrix = np.array([[np.cos(theta / 2), (-1j) * np.sin(theta / 2)],
                                          [(-1j) * np.sin(theta / 2), np.cos(theta / 2)]])

                    unitary_list.append(rx_matrix)
                    targets.append(pair.target + extra_q)

                    if pair.target == 1:
                        Cldmcsu.cldmcsu(gate_circ, unitary_list, control_list, targets)
                else:
                    gate_circ.crx(signal * np.pi / param, pair.control + extra_q, pair.target + extra_q)

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
    def ldmcu_approx(circuit, unitary, controls, target, error, ctrl_state=None):
        '''
        Append LdmcuApprox to the circuit
        '''
        circuit.append(
            LdmcuApprox(unitary, len(controls), error, ctrl_state=ctrl_state),
            [*controls, target]
        )


LdmcuApprox._apply_ctrl_state = apply_ctrl_state
