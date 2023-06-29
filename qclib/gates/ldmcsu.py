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

from typing import Union, List

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate, RYGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import OneQubitEulerDecomposer
from qiskit.circuit import Gate, Qubit

from qclib.gates.mcx import LinearMcx, McxVchainDirty
from qclib.gates.util import check_su2, apply_ctrl_state, isclose


# pylint: disable=protected-access


class Ldmcsu(Gate):
    """
    Linear depth Multi-Controlled Gate for Special Unitary
    ------------------------------------------------

    Multicontrolled gate decomposition with linear cost.
    `unitary` must be a SU(2) matrix.
    """

    def __init__(self, unitary, num_controls, num_target_qubit=1, ctrl_state: str = None):

        check_su2(unitary)
        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(num_target_qubit)
        self.num_controls = num_controls + 1
        self.ctrl_state = ctrl_state

        super().__init__("ldmcsu", self.num_controls, [], "ldmcsu")

    def _define(self):

        self.definition = QuantumCircuit(self.controls, self.target)

        is_main_diag_real = isclose(self.unitary[0, 0].imag, 0.0) and isclose(
            self.unitary[1, 1].imag, 0.0
        )
        is_secondary_diag_real = isclose(self.unitary[0, 1].imag, 0.0) and isclose(
            self.unitary[1, 0].imag, 0.0
        )

        if not is_main_diag_real and not is_secondary_diag_real:
            # U = V D V^-1, where the entries of the diagonal D are the eigenvalues
            # `eig_vals` of U and the column vectors of V are the eigenvectors
            # `eig_vecs` of U. These columns are orthonormal and the main diagonal
            # of V is real-valued.
            eig_vals, eig_vecs = np.linalg.eig(self.unitary)

            x_vecs, z_vecs = self._get_x_z(eig_vecs)
            x_vals, z_vals = self._get_x_z(np.diag(eig_vals))

            self.half_linear_depth_mcv(
                x_vecs,
                z_vecs,
                self.controls,
                self.target,
                self.ctrl_state,
                inverse=True,
            )
            self.linear_depth_mcv(
                x_vals,
                z_vals,
                self.controls,
                self.target,
                self.ctrl_state,
                general_su2_optimization=True,
            )
            self.half_linear_depth_mcv(
                x_vecs, z_vecs, self.controls, self.target, self.ctrl_state
            )

        else:
            x_value, z_value = self._get_x_z(self.unitary)

            if not is_secondary_diag_real:
                self.definition.h(self.target)

            self.linear_depth_mcv(x_value, z_value, self.controls, self.target, self.ctrl_state)

            if not is_secondary_diag_real:
                self.definition.h(self.target)

    @staticmethod
    def _get_x_z(su2):
        is_secondary_diag_real = isclose(su2[0, 1].imag, 0.0) and isclose(
            su2[1, 0].imag, 0.0
        )

        if is_secondary_diag_real:
            x_value = su2[0, 1]
            z_value = su2[1, 1]
        else:
            x_value = -su2[0, 1].real
            z_value = su2[1, 1] - su2[0, 1].imag * 1.0j

        return x_value, z_value

    @staticmethod
    def _compute_gate_a(x_value, z_value):
        if x_value == 0:
            alpha = (z_value + 0j) ** (1 / 4)
            beta = 0.0
        else:
            alpha_r = np.sqrt((np.sqrt((z_value.real + 1.0) / 2.0) + 1.0) / 2.0)
            alpha_i = z_value.imag / (
                    2.0 * np.sqrt((z_value.real + 1.0) *
                                  (np.sqrt((z_value.real + 1.0) / 2.0) + 1.0))
            )
            alpha = alpha_r + 1.0j * alpha_i
            beta = x_value / (
                    2.0 * np.sqrt((z_value.real + 1.0) *
                                  (np.sqrt((z_value.real + 1.0) / 2.0) + 1.0))
            )
        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])
        return s_op

    # x_values = [x_value_1, ..., x_value_p]
    # z_values = [z_value_1, ..., z_value_p]
    def linear_depth_mcv_multi_target(
            self,
            x_values,
            z_values,
            controls: Union[QuantumRegister, List[Qubit]],
            target: Qubit,
            ctrl_state: str = None,
            general_su2_optimization=False,
    ):
        """
                Implementa p operadores $U \in SU2$ multicontrolados
                Theorem 1 - https://arxiv.org/pdf/2302.06377.pdf
        """
        gates_a = []
        num_target_qubit = len(x_values)
        for p in range(num_target_qubit):
            op_a = Ldmcsu._compute_gate_a(x_values[p], z_values[p])
            gates_a.append(UnitaryGate(op_a))

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        mcx_k1 = McxVchainDirty(k_1, num_target_qubit=num_target_qubit, ctrl_state=ctrl_state_k_1).definition
        mcx_k2 = McxVchainDirty(k_2, num_target_qubit=num_target_qubit, ctrl_state=ctrl_state_k_2).definition

        # 1 - fazer append mcx_k1

        # 2 - fazer append dos p operadores A
        for p, t in range(num_target_qubit):
            self.definition.append(gates_a[p], [target][p])

        # 3 - fazer append mcx_k2

        # 4 - fazer append dos p operadores A^\dagger
        for p in range(num_target_qubit):
            self.definition.append(gates_a[p].inverse(), [target])

        # 1 - fazer append mcx_k1

        # 2 - fazer append dos p operadores A
        for p in range(num_target_qubit):
            self.definition.append(gates_a[p], [target])

        # 3 - fazer append mcx_k2

        # 4 - fazer append dos p operadores A^\dagger
        for p in range(num_target_qubit):
            self.definition.append(gates_a[p].inverse(), [target])





    def linear_depth_mcv(
            # função a ser modificada
            self,
            x_value,
            z_value,
            controls: Union[QuantumRegister, List[Qubit]],
            target: Qubit,
            ctrl_state: str = None,
            general_su2_optimization=False,
    ):
        """
        Theorem 1 - https://arxiv.org/pdf/2302.06377.pdf
        """
        # S gate definition
        op_a = Ldmcsu._compute_gate_a(x_value, z_value)
        gate_a = UnitaryGate(op_a)

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if not general_su2_optimization:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(
                mcx_1, controls[:k_1] + controls[k_1: 2 * k_1 - 2] + [target]
            )
        self.definition.append(gate_a, [target])

        mcx_2 = McxVchainDirty(
            k_2, ctrl_state=ctrl_state_k_2, action_only=general_su2_optimization
        ).definition
        self.definition.append(
            mcx_2.inverse(), controls[k_1:] + controls[k_1 - k_2 + 2: k_1] + [target]
        )
        self.definition.append(gate_a.inverse(), [target])

        mcx_3 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
        self.definition.append(
            mcx_3, controls[:k_1] + controls[k_1: 2 * k_1 - 2] + [target]
        )
        self.definition.append(gate_a, [target])

        mcx_4 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
        self.definition.append(
            mcx_4, controls[k_1:] + controls[k_1 - k_2 + 2: k_1] + [target]
        )
        self.definition.append(gate_a.inverse(), [target])

    def half_linear_depth_mcv(
            self,
            x_value,
            z_value,
            controls: Union[QuantumRegister, List[Qubit]],
            target: Qubit,
            ctrl_state: str = None,
            inverse: bool = False,
    ):
        """
        Theorem 4 - https://arxiv.org/pdf/2302.06377.pdf
        """

        alpha_r = np.sqrt((z_value.real + 1.0) / 2.0)
        alpha_i = z_value.imag / np.sqrt(2 * (z_value.real + 1.0))
        alpha = alpha_r + 1.0j * alpha_i

        beta = x_value / np.sqrt(2 * (z_value.real + 1.0))

        s_op = np.array([[alpha, -np.conj(beta)], [beta, np.conj(alpha)]])

        # S gate definition
        s_gate = UnitaryGate(s_op)

        # Hadamard equivalent definition
        h_gate = UnitaryGate(np.array([[-1, 1], [1, 1]]) * 1 / np.sqrt(2))

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if inverse:
            self.definition.h(target)

            self.definition.append(s_gate, [target])
            mcx_2 = McxVchainDirty(
                k_2, ctrl_state=ctrl_state_k_2, action_only=True
            ).definition
            self.definition.append(
                mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2: k_1] + [target]
            )

            self.definition.append(s_gate.inverse(), [target])

            self.definition.append(h_gate, [target])

        else:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(
                mcx_1, controls[:k_1] + controls[k_1: 2 * k_1 - 2] + [target]
            )
            self.definition.append(h_gate, [target])

            self.definition.append(s_gate, [target])

            mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
            self.definition.append(
                mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2: k_1] + [target]
            )
            self.definition.append(s_gate.inverse(), [target])

            self.definition.h(target)

    @staticmethod
    def ldmcsu(
            circuit,
            unitary,
            controls: Union[QuantumRegister, List[Qubit]],
            target: Qubit,
            ctrl_state: str = None,
    ):
        """
        Apply multi-controlled SU(2)
        https://arxiv.org/abs/2302.06377
        """
        circuit.append(
            Ldmcsu(unitary, len(controls), ctrl_state=ctrl_state), [*controls, target]
        )


class LdMcSpecialUnitary(Gate):
    """
    Linear-depth Multicontrolled Special Unitary
    --------------------------------------------

    Implements the gate decompostion of any gate in SU(2) with linear depth (Ld)
    presented in Lemma 7.9 in Barenco et al., 1995 (arXiv:quant-ph/9503016)
    with optimizations from Theorem 5 of Iten et al., 2016 (arXiv:1501.06911)
    """

    def __init__(self, unitary, num_controls, ctrl_state=None):

        if not check_su2(unitary):
            raise Exception("Operator must be in SU(2)")

        self.unitary = np.array(unitary, dtype=complex)

        if num_controls > 0:
            self.control_qubits = QuantumRegister(num_controls)
        else:
            self.control_qubits = []

        self.target_qubit = QuantumRegister(1)
        self.num_qubits = num_controls + 1
        self.ctrl_state = ctrl_state

        if self.ctrl_state is None:
            self.ctrl_state = "1" * num_controls

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

        a_gate = UnitaryGate(a_matrix, label="A")
        b_gate = UnitaryGate(b_matrix, label="B")
        c_gate = UnitaryGate(c_matrix, label="C")

        return a_gate, b_gate, c_gate

    def _define(self):
        self.definition = QuantumCircuit(self.control_qubits, self.target_qubit)

        if len(self.control_qubits) > 0:
            self._apply_ctrl_state()

            theta, phi, lamb, _ = OneQubitEulerDecomposer._params_zyz(self.unitary)

            a_gate, b_gate, c_gate = LdMcSpecialUnitary.get_abc_operators(
                phi, theta, lamb
            )

            self._apply_abc(a_gate, b_gate, c_gate)

            self._apply_ctrl_state()
        else:
            self.unitary(self.unitary, self.target_qubit)

    def _apply_abc(self, a_gate: UnitaryGate, b_gate: UnitaryGate, c_gate: UnitaryGate):
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
            theta_a, phi_a, lam_a, _ = OneQubitEulerDecomposer._params_zyz(
                a_gate.to_matrix()
            )
            theta_b, phi_b, lam_b, _ = OneQubitEulerDecomposer._params_zyz(
                b_gate.to_matrix()
            )
            theta_c, phi_c, lam_c, _ = OneQubitEulerDecomposer._params_zyz(
                c_gate.to_matrix()
            )
            a_a, b_a, c_a = LdMcSpecialUnitary.get_abc_operators(phi_a, theta_a, lam_a)
            a_b, b_b, c_b = LdMcSpecialUnitary.get_abc_operators(phi_b, theta_b, lam_b)
            a_c, b_c, c_c = LdMcSpecialUnitary.get_abc_operators(phi_c, theta_c, lam_c)

            # definition of left mcx, which will also be inverted as the right mcx
            mcx_gate = LinearMcx(
                len(self.control_qubits[:-1]), action_only=action_only
            ).definition

            # decomposed controlled C
            self.definition.unitary(c_c, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_c, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_c, self.target_qubit)

            self.definition.append(
                mcx_gate, self.control_qubits[:-1] + [self.target_qubit] + [ancilla]
            )

            # decomposed controlled B
            self.definition.unitary(c_b, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_b, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_b, self.target_qubit)

            self.definition.append(
                mcx_gate.inverse(),
                self.control_qubits[:-1] + [self.target_qubit] + [ancilla],
            )

            # decomposed A
            self.definition.unitary(c_a, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(b_a, self.target_qubit)
            self.definition.cx(ancilla, self.target_qubit)
            self.definition.unitary(a_a, self.target_qubit)

    @staticmethod
    def ldmcsu(circuit, unitary, controls, target, ctrl_state=None):
        """
        Linear-depth Multicontrolled Special Unitary
        --------------------------------------------

        Implements the gate decompostion of any gate in SU(2) with linear depth (Ld)
        presented in Lemma 7.9 in Barenco et al., 1995 (arXiv:quant-ph/9503016)
        with optimizations from Theorem 5 of Iten et al., 2016 (arXiv:1501.06911)
        """
        circuit.append(
            LdMcSpecialUnitary(unitary, len(controls), ctrl_state), [*controls, target]
        )


LdMcSpecialUnitary._apply_ctrl_state = apply_ctrl_state
