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
from qiskit.extensions import UnitaryGate
from qiskit.circuit import Gate, Qubit
from qclib.gates.ldmcsu import Ldmcsu
from qclib.gates.mcx import McxVchainDirty
from qclib.gates.util import check_su2, isclose

# pylint: disable=protected-access


class MultiTargetMcSU2(Gate):
    """
    Multi-target Multi-Controlled Gate for Special Unitary
    ------------------------------------------------

    Multicontrolled gate decomposition with linear cost.
    """

    def __init__(self, unitaries, num_controls, num_target=1, ctrl_state: str = None):
        """
        Parameters
        ----------
        unitaries (list): List of SU(2) matrices
        num_controls (int): Number of control gastes
        num_target (int): Number of target gates
        ctrl_state (str): Control state in decimal or as a bitstring
        """

        if isinstance(unitaries, list):
            for unitary in unitaries:
                check_su2(unitary)
        else:
            check_su2(unitaries)

        self.unitaries = unitaries
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(num_target)
        self.num_controls = num_controls + 1
        self.ctrl_state = ctrl_state

        super().__init__("ldmcsu", self.num_controls, [], "ldmcsu")

    def _define(self):
        if isinstance(self.unitaries, list):
            self.definition = QuantumCircuit(self.controls, self.target)

            is_main_diags_real = []
            is_secondary_diags_real = []
            for unitary in self.unitaries:
                is_main_diags_real.append(
                    isclose(unitary[0, 0].imag, 0.0)
                    and isclose(unitary[1, 1].imag, 0.0)
                )

                is_secondary_diags_real.append(
                    isclose(unitary[0, 1].imag, 0.0)
                    and isclose(unitary[1, 0].imag, 0.0)
                )

            for idx, unitary in enumerate(self.unitaries):
                if not is_secondary_diags_real[idx] and is_main_diags_real[idx]:
                    self.definition.h(self.target[idx])

            self.clinear_depth_mcv()

            for idx, unitary in enumerate(self.unitaries):
                if not is_secondary_diags_real[idx] and is_main_diags_real[idx]:
                    self.definition.h(self.target[idx])

        else:
            self.definition = Ldmcsu(self.unitaries, self.num_controls)

    def clinear_depth_mcv(self, general_su2_optimization=False):
        """
        Multi-target version of Theorem 1 - https://arxiv.org/pdf/2302.06377.pdf
        """
        # S gate definition
        gates_a = self._s_gate_definition(self.unitaries)

        num_ctrl = len(self.controls)
        target_size = len(self.target)


        k_1 = int(np.ceil(num_ctrl / 2.0))
        k_2 = int(np.floor(num_ctrl / 2.0))

        ctrl_state_k_1 = None
        ctrl_state_k_2 = None

        if self.ctrl_state is not None:
            ctrl_state_k_1 = self.ctrl_state[::-1][:k_1][::-1]
            ctrl_state_k_2 = self.ctrl_state[::-1][k_1:][::-1]

        if not general_su2_optimization:
            mcx_1 = McxVchainDirty(
                k_1, num_target_qubit=target_size, ctrl_state=ctrl_state_k_1
            ).definition
            self.definition.append(
                mcx_1,
                self.controls[:k_1] + self.controls[k_1 : 2 * k_1 - 2] + [*self.target],
            )

        for idx, gate_a in enumerate(gates_a):
            self.definition.append(gate_a, [num_ctrl + idx])

        mcx_2 = McxVchainDirty(
            k_2,
            num_target_qubit=target_size,
            ctrl_state=ctrl_state_k_2,
            action_only=general_su2_optimization,
        ).definition

        self.definition.append(
            mcx_2.inverse(),
            self.controls[k_1:] + self.controls[k_1 - k_2 + 2 : k_1] + [*self.target],
        )

        for idx, gate_a in enumerate(gates_a):
            self.definition.append(gate_a.inverse(), [num_ctrl + idx])

        mcx_3 = McxVchainDirty(
            k_1, num_target_qubit=target_size, ctrl_state=ctrl_state_k_1
        ).definition

        self.definition.append(
            mcx_3,
            self.controls[:k_1] + self.controls[k_1 : 2 * k_1 - 2] + [*self.target],
        )

        for idx, gate_a in enumerate(gates_a):
            self.definition.append(gate_a, [num_ctrl + idx])

        mcx_4 = McxVchainDirty(
            k_2, num_target_qubit=target_size, ctrl_state=ctrl_state_k_2
        ).definition

        self.definition.append(
            mcx_4,
            self.controls[k_1:] + self.controls[k_1 - k_2 + 2 : k_1] + [*self.target],
        )

        for idx, gate_a in enumerate(gates_a):
            self.definition.append(gate_a.inverse(), [num_ctrl + idx])

    def _s_gate_definition(self, su2_unitaries):
        gates_a = []
        for su2_unitary in su2_unitaries:
            x_value, z_value = Ldmcsu._get_x_z(su2_unitary)
            op_a = Ldmcsu._compute_gate_a(x_value, z_value)
            gates_a.append(UnitaryGate(op_a))
        return gates_a

    @staticmethod
    def cldmcsu(
        circuit,
        unitary,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str = None,
    ):
        """
        Multi-target version of multi-controlled SU(2)
        https://arxiv.org/abs/2302.06377
        """
        if isinstance(unitary, list):
            num_target = len(unitary)
            circuit.append(
                MultiTargetMcSU2(unitary, len(controls), num_target=num_target).definition,
                [*controls, *target],
            )
        else:
            circuit.append(
                Ldmcsu(unitary, len(controls), ctrl_state=ctrl_state),
                [*controls, target],
            )
