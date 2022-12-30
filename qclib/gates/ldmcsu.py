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
Linear-depth Multicontrolled Special Unitary
"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate, RYGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.circuit import Gate

from .mcx_gate import LinearMcx
from .mcg import _check_su2
from ._utils import _apply_ctrl_state

# pylint: disable=protected-access

class LdMcSpecialUnitary(Gate):
    """
        Linear-depth Multicontrolled Special Unitary
        --------------------------------------------

        Implements the gate decompostion of any gate in SU(2) with linear depth (Ld)
        presented in Lemma 7.9 in Barenco et al., 1995 (arXiv:quant-ph/9503016)
        with optimizations from Theorem 5 of Iten et al., 2016 (arXiv:1501.06911)
    """
    def __init__(self, unitary, num_controls,  ctrl_state=None):

        if not _check_su2(unitary):
            raise Exception("Operator must be in SU(2)")

        self.unitary = unitary
        self.control_qubits = QuantumRegister(num_controls)
        self.target_qubit = QuantumRegister(1)
        self.num_qubits = num_controls + 1
        self.ctrl_state = ctrl_state

        if self.ctrl_state is None:
            self.ctrl_state = '1' * num_controls

        super().__init__("ldmc_su2", self.num_qubits, [], "LdMcSu2")

    @staticmethod
    def get_abc_operators(beta, gamma, delta):
        """
        Creates A,B and C matrices such that
        ABC = I
        """
        # A
        a_rz = RZGate(beta).to_matrix()
        a_ry = RYGate(gamma / 2).to_matrix()
        a_matrix = a_rz.dot(a_ry)

        # B
        b_ry = RYGate(-gamma / 2).to_matrix()
        b_rz = RZGate(-(delta + beta) / 2).to_matrix()
        b_matrix = b_ry.dot(b_rz)

        # C
        c_matrix = RZGate((delta - beta) / 2).to_matrix()

        a_gate = UnitaryGate(a_matrix, label='A')
        b_gate = UnitaryGate(b_matrix, label='B')
        c_gate = UnitaryGate(c_matrix, label='C')

        return a_gate, b_gate, c_gate

    def _define(self):
        self.definition = QuantumCircuit(self.control_qubits, self.target_qubit)

        if len(self.control_qubits) > 0:
            self._apply_ctrl_state()

            theta, phi, lamb, _ = OneQubitEulerDecomposer._params_zyz(self.unitary)

            a_gate, b_gate, c_gate = LdMcSpecialUnitary.get_abc_operators(phi, theta, lamb)

            self._apply_abc(
                a_gate, b_gate, c_gate
            )

            self._apply_ctrl_state()
        else:
            self.unitary(self.unitary, self.target_qubit)

    def _apply_abc(
        self,
        a_gate: UnitaryGate,
        b_gate: UnitaryGate,
        c_gate: UnitaryGate
    ):
        """
            Applies ABC matrices to the quantum circuit according to theorem 5
            of Iten et al. 2016 (arXiv:1501.06911).
            Parameters
            ----------
                a_gate, b_gate and c_gate expceted to be special unitary gates
        """

        if len(self.control_qubits) < 3:
            self.definition.append(c_gate, [self.target_qubit])
            self.definition.mcx(self.control_qubits, self.target_qubit)
            self.definition.append(b_gate, [self.target_qubit])
            self.definition.mcx(self.control_qubits, self.target_qubit)
            self.definition.append(a_gate, [self.target_qubit])
        else:
            ancilla = self.control_qubits[-1]
            action_only = True

            if len(self.control_qubits) < 6:
                action_only = False

            # decompose A, B and C to use their optimized controlled versions
            theta_a, phi_a, lam_a, _ = OneQubitEulerDecomposer._params_zyz(a_gate.to_matrix())
            theta_b, phi_b, lam_b, _ = OneQubitEulerDecomposer._params_zyz(b_gate.to_matrix())
            theta_c, phi_c, lam_c, _ = OneQubitEulerDecomposer._params_zyz(c_gate.to_matrix())
            a_a, b_a, c_a = LdMcSpecialUnitary.get_abc_operators(phi_a, theta_a, lam_a)
            a_b, b_b, c_b = LdMcSpecialUnitary.get_abc_operators(phi_b, theta_b, lam_b)
            a_c, b_c, c_c = LdMcSpecialUnitary.get_abc_operators(phi_c, theta_c, lam_c)

            # definition of left mcx, which will also be inverted as the right mcx
            mcx_gate = LinearMcx(len(self.control_qubits[:-1]), action_only=action_only).definition

            # decomposed controlled C
            self.definition.unitary(c_c, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_c, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_c, self.target_qubit)

            self.definition.append(
                mcx_gate,
                self.control_qubits[:-1] + [self.target_qubit] + [ancilla]
            )

            # decomposed controlled B
            self.definition.unitary(c_b, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_b, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_b, self.target_qubit)

            self.definition.append(
                mcx_gate.inverse(),
                self.control_qubits[:-1] + [self.target_qubit] + [ancilla]
            )

            # decomposed A
            self.definition.unitary(c_a, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_a, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_a, self.target_qubit)

LdMcSpecialUnitary._apply_ctrl_state = _apply_ctrl_state
