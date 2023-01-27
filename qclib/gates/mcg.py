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

"""Multicontrolled gate decompositions for unitaries in U(2) and SU(2)"""

from typing import Union, List

from cmath import isclose
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit import Gate
from .mcx_gate import McxVchainDirty
from .mc_gate import mc_gate
from ._utils import _check_u2, _check_su2, _u2_to_su2


# pylint: disable=maybe-no-member
# pylint: disable=protected-access


class Mcg(Gate):
    """
    Selects the most cost-effective multicontrolled gate decomposition.
    """

    def __init__(
        self,
        unitary,
        num_controls,
        ctrl_state: str=None,
        up_to_diagonal: bool=False
    ):

        _check_u2(unitary)
        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(1)
        self.num_qubits = num_controls + 1
        self.ctrl_state = ctrl_state
        self.up_to_diagonal = up_to_diagonal

        super().__init__("lmcgsu", self.num_qubits, [], "lmcgsu")

    def _define(self):

        self.definition = QuantumCircuit(self.controls, self.target)

        num_ctrl = len(self.controls)
        if num_ctrl == 0:
            self.definition.unitary(self.unitary, [self.target])
        elif num_ctrl == 1:
            u_gate = QuantumCircuit(1)
            u_gate.unitary(self.unitary, 0)
            self.definition.append(
                u_gate.control(num_ctrl, ctrl_state=self.ctrl_state),
                [*self.controls, self.target]
            )
        else:
            if _check_su2(self.unitary):
                LMCGSU.lmcgsu(
                    self.definition,
                    self.unitary,
                    self.controls,
                    self.target,
                    ctrl_state=self.ctrl_state
                )
            else:
                if self.up_to_diagonal:
                    su_2, _ = _u2_to_su2(self.unitary)
                    self.mcg(su_2, self.controls, self.target, self.ctrl_state)
                else:
                    mc_gate(self.unitary, self.definition, self.controls[:], self.target[0], self.ctrl_state)
    
    @staticmethod
    def mcg(
        circuit,
        unitary,
        controls,
        target,
        ctrl_state: str = None
    ):
        circuit.append(
            Mcg(unitary, len(controls), ctrl_state=ctrl_state), 
            [*controls, target]
        )


class LMCGSU(Gate):
    """
    Linear Multi-Controlled Gate for Special Unitary
    ------------------------------------------------

    Multicontrolled gate decomposition with linear cost.
    `unitary` must be a SU(2) matrix with at least one real diagonal.
    """

    def __init__(
        self,
        unitary,
        num_controls,
        ctrl_state: str=None
    ):

        _check_su2(unitary)
        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(1)
        self.num_controls = num_controls + 1
        self.ctrl_state = ctrl_state

        super().__init__("lmcgsu", self.num_controls, [], "lmcgsu")

    def _define(self):

        self.definition = QuantumCircuit(self.controls, self.target)

        is_main_diag_real = isclose(self.unitary[0, 0].imag, 0.0) and \
                            isclose(self.unitary[1, 1].imag, 0.0)
        is_secondary_diag_real = isclose(self.unitary[0,1].imag, 0.0) and \
                                  isclose(self.unitary[1,0].imag, 0.0)

        if  not is_main_diag_real and not is_secondary_diag_real:
            # U = V D V^-1, where the entries of the diagonal D are the eigenvalues
            # `eig_vals` of U and the column vectors of V are the eigenvectors
            # `eig_vecs` of U. These columns are orthonormal and the main diagonal
            # of V is real-valued.
            eig_vals, eig_vecs = np.linalg.eig(self.unitary)

            x_vecs, z_vecs = self._get_x_z(eig_vecs)
            x_vals, z_vals = self._get_x_z(np.diag(eig_vals))

            self.half_linear_depth_mcv(
                x_vecs, z_vecs, self.controls, self.target, self.ctrl_state, inverse=True
            )
            self.linear_depth_mcv(
                x_vals,
                z_vals,
                self.controls,
                self.target,
                self.ctrl_state,
                general_su2_optimization=True
            )
            self.half_linear_depth_mcv(
                x_vecs, z_vecs, self.controls, self.target, self.ctrl_state
            )

        else:
            x, z = self._get_x_z(self.unitary)

            if not is_secondary_diag_real:
                self.definition.h(self.target)

            self.linear_depth_mcv(
                x,
                z,
                self.controls,
                self.target,
                self.ctrl_state
            )

            if not is_secondary_diag_real:
                self.definition.h(self.target)

    @staticmethod
    def _get_x_z(su2):
        is_secondary_diag_real = isclose(su2[0,1].imag, 0.0) and isclose(su2[1,0].imag, 0.0)

        if is_secondary_diag_real:
            x = su2[0,1]
            z = su2[1,1]
        else:
            x = -su2[0,1].real
            z = su2[1,1] - su2[0,1].imag * 1.0j

        return x, z

    def linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None,
        general_su2_optimization=False
    ):

        alpha_r = np.sqrt(
        (np.sqrt((z.real + 1.) / 2.) + 1.) / 2.
        )
        alpha_i = z.imag / (2. * np.sqrt(
            (z.real + 1.) * (np.sqrt((z.real + 1.) / 2.) + 1.)
        ))
        alpha = alpha_r + 1.j * alpha_i
        beta = x / (2. * np.sqrt(
                (z.real + 1.) * (np.sqrt((z.real + 1.) / 2.) + 1.)
            )
        )

        s_op = np.array(
            [[alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]]
        )

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.))
        k_2 = int(np.floor(num_ctrl / 2.))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if not general_su2_optimization:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
        self.definition.append(s_gate, [target])

        mcx_2 = McxVchainDirty(
            k_2, ctrl_state=ctrl_state_k_2, action_only=general_su2_optimization
        ).definition
        self.definition.append(mcx_2.inverse(), controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
        self.definition.append(s_gate.inverse(), [target])

        mcx_3 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
        self.definition.append(mcx_3, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
        self.definition.append(s_gate, [target])

        mcx_4 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
        self.definition.append(mcx_4, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
        self.definition.append(s_gate.inverse(), [target])

    def half_linear_depth_mcv(
        self,
        x,
        z,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None,
        inverse: bool=False
    ):

        alpha_r = np.sqrt((z.real + 1.) / 2.)
        alpha_i = z.imag / np.sqrt(2*(z.real + 1.))
        alpha = alpha_r + 1.j * alpha_i

        beta = x / np.sqrt(2*(z.real + 1.))

        s_op = np.array(
            [[alpha, -np.conj(beta)],
            [beta, np.conj(alpha)]]
        )

        # S gate definition
        s_gate = QuantumCircuit(1)
        s_gate.unitary(s_op, 0)

        # Hadamard equivalent definition
        h_gate = QuantumCircuit(1)
        h_gate.unitary(np.array([[-1, 1], [1, 1]]) * 1/np.sqrt(2), 0)

        num_ctrl = len(controls)
        k_1 = int(np.ceil(num_ctrl / 2.))
        k_2 = int(np.floor(num_ctrl / 2.))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if ctrl_state is not None:
            ctrl_state_k_1 = ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = ctrl_state[::-1][k_1:][::-1]

        if inverse:
            self.definition.h(target)

            self.definition.append(s_gate, [target])
            mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2, action_only=True).definition
            self.definition.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])

            self.definition.append(s_gate.inverse(), [target])

            self.definition.append(h_gate, [target])

        else:
            mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
            self.definition.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
            self.definition.append(h_gate, [target])

            self.definition.append(s_gate, [target])

            mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
            self.definition.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
            self.definition.append(s_gate.inverse(), [target])

            self.definition.h(target)

    @staticmethod
    def lmcgsu(
        circuit,
        unitary,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None
    ):
        circuit.append(
            LMCGSU(unitary, len(controls), ctrl_state=ctrl_state),
            [*controls, target]
        )
