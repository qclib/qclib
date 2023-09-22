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
linear-depth n-qubit controlled X with ancilla
"""

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.circuit.library import C3XGate, C4XGate
from qiskit.circuit import Gate
from qclib.gates.toffoli import Toffoli
from qclib.gates.util import apply_ctrl_state
from qiskit.circuit.library.standard_gates import SXGate
from qiskit.circuit.library import SXdgGate


# pylint: disable=protected-access


class McxVchainDirty(Gate):
    """
    Implementation based on lemma 8 of Iten et al. (2016) arXiv:1501.06911.
    Decomposition of a multicontrolled X gate with at least k <= ceil(n/2) ancilae
    for n as the total number of qubits in the system. It also includes optimizations
    using approximated Toffoli gates up to a diagonal.
    """

    def __init__(
            self,
            num_controls: int,
            num_target_qubit=1,
            ctrl_state=None,
            relative_phase=False,
            action_only=False,
    ):
        """
        Parameters
        ----------
        num_target_qubit
        num_controls
        ctrl_state
        relative_phase
        action_only
        """
        self.control_qubits = QuantumRegister(num_controls)
        self.target_qubits = QuantumRegister(num_target_qubit)
        self.ctrl_state = ctrl_state
        self.relative_phase = relative_phase
        self.action_only = action_only

        num_ancilla = 0
        self.ancilla_qubits = []
        if num_controls - 2 > 0:
            num_ancilla = num_controls - 2
            self.ancilla_qubits = QuantumRegister(num_ancilla)

        super().__init__(
            "mcx_vc_dirty", num_controls + num_ancilla + 1, [], "mcx_vc_dirty"
        )

    def toffoli_multi_target(self, num_targets):
        """"
        """
        self.num_targets = num_targets
        size = self.num_targets + 2
        circuit = QuantumCircuit(size)
        sx_gate = SXGate().control(1)
        sxdg_gate = SXdgGate().control(1)

        for i in range(self.num_targets - 1):
            circuit.cx(size - i - 2, size - i - 1)

        circuit.append(sx_gate, [1, 2])
        circuit.cx(0, 1)
        circuit.append(sxdg_gate, [1, 2])
        circuit.cx(0, 1)
        circuit.append(sx_gate, [0, 2])

        for i in range(self.num_targets - 1):
            circuit.cx(i + 2, i + 3)

        return circuit


    def _define(self):
        self.definition = QuantumCircuit(
            self.control_qubits, self.ancilla_qubits, self.target_qubits
        )

        num_ctrl = len(self.control_qubits)
        num_target = len(self.target_qubits)
        num_ancilla = num_ctrl - 2
        targets_aux = self.target_qubits[0:1] + self.ancilla_qubits[:num_ancilla][::-1]

        self._apply_ctrl_state()
        if num_ctrl < 3:
            for k, _ in enumerate(self.target_qubits):
                self.definition.mcx(
                    control_qubits=self.control_qubits,
                    target_qubit=self.target_qubits[k],
                    mode="noancilla",
                )

        elif not self.relative_phase and num_ctrl == 3 and num_target < 2:
            for k, _ in enumerate(self.target_qubits):
                self.definition.append(
                    C3XGate(), [*self.control_qubits[:], self.target_qubits[k]], []
                )
        else:
            for j in range(2):
                for i, _ in enumerate(self.control_qubits):  # action part
                    if i < num_ctrl - 2:
                        if (
                                targets_aux[i] not in self.target_qubits
                                or self.relative_phase
                        ):
                            # gate cancelling
                            controls = [
                                self.control_qubits[num_ctrl - i - 1],
                                self.ancilla_qubits[num_ancilla - i - 1],
                            ]

                            # cancel rightmost gates of action part
                            # with leftmost gates of reset part
                            if (
                                    self.relative_phase
                                    and targets_aux[i] in self.target_qubits
                                    and j == 1
                            ):
                                self.definition.append(
                                    Toffoli(cancel="left"), [*controls, targets_aux[i]]
                                )
                            else:
                                self.definition.append(
                                    Toffoli(cancel="right"), [*controls, targets_aux[i]]
                                )

                        else:

                            control_1 = self.control_qubits[num_ctrl - i - 1]
                            control_2 = self.ancilla_qubits[num_ancilla - i - 1]
                            targets = self.target_qubits
                            num_targets = len(targets)
                            self.definition.append(
                                self.toffoli_multi_target(num_targets),
                                [control_1, control_2, *targets]
                            )

                    else:
                        controls = [
                            self.control_qubits[num_ctrl - i - 2],
                            self.control_qubits[num_ctrl - i - 1],
                        ]

                        self.definition.append(Toffoli(), [*controls, targets_aux[i]])

                        break

                for i, _ in enumerate(self.ancilla_qubits[1:]):  # reset part
                    controls = [self.control_qubits[2 + i], self.ancilla_qubits[i]]
                    self.definition.append(
                        Toffoli(cancel="left"), [*controls, self.ancilla_qubits[i + 1]]
                    )

                if self.action_only:

                    control_1 = self.control_qubits[-1]
                    control_2 = self.ancilla_qubits[-1]
                    targets = self.target_qubits
                    num_targets = len(targets)
                    self.definition.append(
                        self.toffoli_multi_target(num_targets),
                        [control_1, control_2, *targets]
                    )
                    # for target in self.target_qubits:
                    #     self.definition.ccx(
                    #         control_qubit1=self.control_qubits[-1],
                    #         control_qubit2=self.ancilla_qubits[-1],
                    #         target_qubit=target,
                    #     )

                    break

        self._apply_ctrl_state()

    @staticmethod
    def mcx_vchain_dirty(
            circuit,
            controls=None,
            target=None,
            ctrl_state=None,
            relative_phase=False,
            action_only=False,
    ):
        circuit.append(
            McxVchainDirty(len(controls), ctrl_state, relative_phase, action_only),
            [*controls, target],
        )


McxVchainDirty._apply_ctrl_state = apply_ctrl_state


class LinearMcx(Gate):
    """
    Implementation based on lemma 9 of Iten et al. (2016) arXiv:1501.06911.
    Decomposition of a multicontrolled X gate with a dirty ancilla by splitting
    it into two sequences of two alternating multicontrolled X gates on
    k1 = ceil((n+1)/2) and k2 = floor((n+1)/2) qubits. For n the total
    number of qubits in the system. Where it also reuses some optimizations available
    """

    def __init__(self, num_controls, ctrl_state=None, action_only=False):
        self.action_only = action_only
        self.ctrl_state = ctrl_state

        num_qubits = num_controls + 2

        self.control_qubits = list(range(num_qubits - 2))
        self.target_qubit = (num_qubits - 2,)
        self.ancilla_qubit = num_qubits - 1

        super().__init__("linear_mcx", num_controls + 2, [], "mcx")

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits)

        self._apply_ctrl_state()
        if self.num_qubits < 5:
            self.definition.mcx(
                control_qubits=self.control_qubits,
                target_qubit=self.target_qubit,
                mode="noancilla",
            )
        elif self.num_qubits == 5:
            self.definition.append(
                C3XGate(), [*self.control_qubits[:], self.target_qubit], []
            )
        elif self.num_qubits == 6:
            self.definition.append(
                C4XGate(), [*self.control_qubits[:], self.target_qubit], []
            )
        elif self.num_qubits == 7:
            self.definition.append(
                C3XGate(), [*self.control_qubits[:3], self.ancilla_qubit], []
            )
            self.definition.append(
                C3XGate(),
                [*self.control_qubits[3:], self.ancilla_qubit, self.target_qubit],
                [],
            )
            self.definition.append(
                C3XGate(), [*self.control_qubits[:3], self.ancilla_qubit], []
            )
            self.definition.append(
                C3XGate(),
                [*self.control_qubits[3:], self.ancilla_qubit, self.target_qubit],
                [],
            )
        else:
            # split controls to halve the number of qubits used for each mcx
            num_ctrl = len(self.control_qubits)
            k_2 = int(np.ceil(self.num_qubits / 2.0))
            k_1 = num_ctrl - k_2 + 1
            first_gate = McxVchainDirty(k_1, relative_phase=True).definition
            second_gate = McxVchainDirty(k_2).definition
            self.definition.append(
                first_gate,
                self.control_qubits[:k_1]
                + self.control_qubits[k_1: k_1 + k_1 - 2]
                + [self.ancilla_qubit],
            )

            self.definition.append(
                second_gate,
                [*self.control_qubits[k_1:], self.ancilla_qubit]
                + self.control_qubits[k_1 - k_2 + 2: k_1]
                + [self.target_qubit],
            )

            self.definition.append(
                first_gate,
                self.control_qubits[:k_1]
                + self.control_qubits[k_1: k_1 + k_1 - 2]
                + [self.ancilla_qubit],
            )

            last_gate = McxVchainDirty(k_2, action_only=self.action_only).definition
            self.definition.append(
                last_gate,
                [*self.control_qubits[k_1:], self.ancilla_qubit]
                + self.control_qubits[k_1 - k_2 + 2: k_1]
                + [self.target_qubit],
            )

        self._apply_ctrl_state()

    @staticmethod
    def mcx(circuit, controls=None, target=None, ctrl_state=None, action_only=False):
        circuit.append(
            LinearMcx(len(controls), ctrl_state, action_only), [*controls, target]
        )


LinearMcx._apply_ctrl_state = apply_ctrl_state
