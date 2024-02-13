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
Quadratic-depth Multicontrolled Special Unitary
"""

from typing import Union, List
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit import Gate
from numpy import sqrt, outer
from numpy.linalg import eig

from .mcx import LinearMcx
from .util import check_u2


# pylint: disable=protected-access


class Qdmcu(Gate):
    """
    Quandratic Depth Multi-Controlled Unitary
    -----------------------------------------

    Implements gate decomposition of a munticontrolled operator in U(2) according to
    Theorem 4 of Iten et al. (2016) arXiv:1501.06911.
    """

    def __init__(
        self,
        unitary,
        num_controls,
        ctrl_state: str=None
    ):
        """
        Parameters
        ----------
        unitary    : numpy.ndarray 2 x 2 unitary matrix
        controls   : Either qiskit.QuantumRegister or list of qiskit.Qubit containing the
                    qubits to be used as control gates.
        target     : qiskit.Qubit on wich the unitary operation is to be applied
        ctrl_state : String of binary digits describing the basis state used as control
        """
        check_u2(unitary)

        self.unitary = unitary
        self.controls = QuantumRegister(num_controls)
        self.target = QuantumRegister(1)
        self.num_qubits = num_controls + 1
        self.ctrl_state = ctrl_state
        super().__init__("qdmcu", self.num_qubits, [], "qdmcu")

    def _define(self):
        self.definition = QuantumCircuit(self.controls, self.target)
        num_ctrl = len(self.controls)

        if num_ctrl == 1:
            u_gate = QuantumCircuit(1)
            u_gate.unitary(self.unitary, 0)
            self.definition.append(
                u_gate.control(
                    num_ctrl_qubits=num_ctrl,
                    ctrl_state=self.ctrl_state
                ),
                [*self.controls, self.target]
            )
        else:
            if self.ctrl_state is None:
                self.ctrl_state = '1' * num_ctrl

            # Notice that `ctrl_state`` is reversed with respect to `controls``.
            v_op = Qdmcu.custom_sqrtm(self.unitary)

            v_gate = QuantumCircuit(1, name="V")
            v_gate.unitary(v_op, 0)

            v_gate_dag = QuantumCircuit(1, name="V^dag")
            v_gate_dag.unitary(v_op.T.conj(), 0)

            linear_mcx_gate = LinearMcx(
                                num_controls=num_ctrl-1,
                                ctrl_state=self.ctrl_state[1:],
                                action_only=True
                            ).definition

            self.definition.append(
                v_gate.control(1, ctrl_state=self.ctrl_state[:1]),
                [self.controls[-1], self.target]
            )
            self.definition.append(
                linear_mcx_gate,
                [*self.controls[:-1], self.controls[-1], self.target]
            )
            self.definition.append(
                v_gate_dag.control(1, ctrl_state=self.ctrl_state[:1]),
                [self.controls[-1], self.target]
            )
            self.definition.append(
                linear_mcx_gate.inverse(),
                [*self.controls[:-1], self.controls[-1], self.target]
            )

            self.qdmcu(
                self.definition,
                v_op,
                self.controls[:-1],
                self.target,
                self.ctrl_state[1:]
            )

    @staticmethod
    def custom_sqrtm(unitary):
        eig_vals, eig_vecs = eig(unitary)
        first_eig = sqrt(eig_vals[0]) * outer(eig_vecs[:, 0], eig_vecs[:, 0].conj())
        second_eig = sqrt(eig_vals[1]) * outer(eig_vecs[:, 1], eig_vecs[:, 1].conj())
        return first_eig + second_eig

    @staticmethod
    def qdmcu(
        circuit,
        unitary,
        controls: Union[QuantumRegister, List[Qubit]],
        target: Qubit,
        ctrl_state: str=None
    ):
        circuit.append(
            Qdmcu(unitary, len(controls), ctrl_state=ctrl_state),
            [*controls, target]
        )
