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

from typing import Union, List, Tuple

from cmath import isclose, phase
import numpy as np

from scipy.linalg import sqrtm

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit.library import RZGate, RYGate, PhaseGate, Diagonal
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import OneQubitEulerDecomposer

from .mcx_gate import McxVchainDirty, LinearMcx
from .mc_gate import mc_gate


# pylint: disable=maybe-no-member
# pylint: disable=protected-access


def mcg(
    self,
    unitary,
    controls: Union[QuantumRegister, List[Qubit]],
    target: Qubit,
    ctrl_state: str=None,
    up_to_diagonal: bool=False
):
    """
    Selects the most cost-effective multicontrolled gate decomposition.
    """
    _check_u2(unitary)

    num_ctrl = len(controls)
    if num_ctrl == 0:
        self.unitary(unitary, [target])
    elif num_ctrl == 1:
        u_gate = QuantumCircuit(1)
        u_gate.unitary(unitary, 0)
        self.append(u_gate.control(num_ctrl, ctrl_state=ctrl_state), [*controls, target])
    else:
        if _check_su2(unitary):
            linear_mcg_su2(self, unitary, controls, target, ctrl_state)
        else:
            if up_to_diagonal:
               su_2, _ = _u2_to_su2(unitary)
               self.mcg(su_2, controls, target, ctrl_state)
            else:
               mc_gate(unitary, self, controls, target, ctrl_state)


def _u2_to_su2(u_2):
    phase_factor = np.conj(np.linalg.det(u_2) ** (-1 / u_2.shape[0]))
    su_2 = u_2 / phase_factor
    return su_2, phase(phase_factor)


def _check_u2(matrix):
    if matrix.shape != (2, 2):
        raise ValueError(
            "The shape of a U(2) matrix must be (2, 2)."
        )
    if not np.allclose(matrix @ np.conj(matrix.T), [[1.0, 0.0],[0.0, 1.0]]):
        raise ValueError(
            "The columns of a U(2) matrix must be orthonormal."
        )

def _check_su2(matrix):
    return isclose(np.linalg.det(matrix), 1.0)


def linear_mcg_su2(
    self,
    unitary,
    controls: Union[QuantumRegister, List[Qubit]],
    target: Qubit,
    ctrl_state: str=None
):
    """
    Multicontrolled gate decomposition with linear cost.
    `unitary` must be a SU(2) matrix with at least one real diagonal.
    """
    is_main_diag_real = isclose(unitary[0, 0].imag, 0.0) and isclose(unitary[1, 1].imag, 0.0)
    is_secondary_diag_real = isclose(unitary[0,1].imag, 0.0) and isclose(unitary[1,0].imag, 0.0)

    if  not is_main_diag_real and not is_secondary_diag_real:
        # U = V D V^-1, where the entries of the diagonal D are the eigenvalues
        # `eig_vals` of U and the column vectors of V are the eigenvectors
        # `eig_vecs` of U. These columns are orthonormal and the main diagonal
        # of V is real-valued.
        eig_vals, eig_vecs = np.linalg.eig(unitary)

        x_vecs, z_vecs = _get_x_z(eig_vecs)
        x_vals, z_vals = _get_x_z(np.diag(eig_vals))

        half_linear_depth_mcv(
            self, x_vecs, z_vecs, controls, target, ctrl_state, inverse=True
        )
        linear_depth_mcv(
            self, x_vals, z_vals, controls, target, ctrl_state, general_su2_optimization=True
        )
        half_linear_depth_mcv(
            self, x_vecs, z_vecs, controls, target, ctrl_state
        )

    else:
        x, z = _get_x_z(unitary)

        if not is_secondary_diag_real:
            self.h(target)

        linear_depth_mcv(
            self,
            x,
            z,
            controls,
            target,
            ctrl_state
        )

        if not is_secondary_diag_real:
            self.h(target)


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
        self.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
    self.append(s_gate, [target])

    mcx_2 = McxVchainDirty(
        k_2, ctrl_state=ctrl_state_k_2, action_only=general_su2_optimization
    ).definition
    self.append(mcx_2.inverse(), controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
    self.append(s_gate.inverse(), [target])

    mcx_3 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
    self.append(mcx_3, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
    self.append(s_gate, [target])

    mcx_4 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
    self.append(mcx_4, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
    self.append(s_gate.inverse(), [target])


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
        self.h(target)

        self.append(s_gate, [target])
        mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2, action_only=True).definition
        self.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])

        self.append(s_gate.inverse(), [target])

        self.append(h_gate, [target])

    else:
        mcx_1 = McxVchainDirty(k_1, ctrl_state=ctrl_state_k_1).definition
        self.append(mcx_1, controls[:k_1] + controls[k_1:2*k_1 - 2] + [target])
        self.append(h_gate, [target])

        self.append(s_gate, [target])

        mcx_2 = McxVchainDirty(k_2, ctrl_state=ctrl_state_k_2).definition
        self.append(mcx_2, controls[k_1:] + controls[k_1 - k_2 + 2:k_1] + [target])
        self.append(s_gate.inverse(), [target])

        self.h(target)
