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

from math import isclose
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import Gate
from .ldmcu import Ldmcu
from .ldmcsu import Ldmcsu
from .util import check_u2, check_su2, u2_to_su2


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

        check_u2(unitary)
        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(1)
        self.num_qubits = num_controls + 1
        self.ctrl_state = ctrl_state
        self.up_to_diagonal = up_to_diagonal

        super().__init__("mcg", self.num_qubits, [], "mcg")

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
            if check_su2(self.unitary):
                Ldmcsu.ldmcsu(
                    self.definition,
                    self.unitary,
                    self.controls,
                    self.target,
                    ctrl_state=self.ctrl_state
                )
            else:
                if self.up_to_diagonal:
                    su_2, _ = u2_to_su2(self.unitary)
                    self.mcg(
                        su_2,
                        self.controls,
                        self.target,
                        self.ctrl_state
                    )
                else:
                    Ldmcu.ldmcu(
                        self.definition,
                        self.unitary,
                        self.controls[:],
                        self.target[0],
                        self.ctrl_state
                    )

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

def cnot_count(unitary, num_controls, up_to_diagonal: bool=False):

    if num_controls >= 7:
        if check_su2(unitary):
            is_main_diag_real = isclose(unitary[0, 0].imag, 0.0) and isclose(
                unitary[1, 1].imag, 0.0
            )
            is_secondary_diag_real = isclose(unitary[0, 1].imag, 0.0) and isclose(
                unitary[1, 0].imag, 0.0
            )

            num_qubits = num_controls + 1
            if is_main_diag_real or is_secondary_diag_real:
                return 16*num_qubits - 40

            if num_qubits % 2 != 0:
                return 20*num_qubits - 38

            return 20*num_qubits - 42
        else:
            return 4*num_qubits**2 - 12*num_qubits + 10

    mcgate = Mcg(
        unitary,
        num_controls,
        up_to_diagonal = up_to_diagonal
    ).definition
    transpiled_mcgate = transpile(mcgate, basis_gates=['u', 'cx'], optimization_level=0)
    return transpiled_mcgate.count_ops().get('cx', 0)
