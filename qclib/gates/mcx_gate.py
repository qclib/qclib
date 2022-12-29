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

def _apply_ctrl_state(self):
    if self.ctrl_state is not None:
        for i, ctrl in enumerate(self.ctrl_state[::-1]):
            if ctrl == '0':
                self.definition.x(self.control_qubits[i])

class McxVchainDirty(Gate):
    """
    Implementation based on lemma 8 of Iten et al. (2016) arXiv:1501.06911.
    Decomposition of a multicontrolled X gate with at least k <= ceil(n/2) ancilae
    for n as the total number of qubits in the system. It also includes optimizations
    using approximated Toffoli gates up to a diagonal.
    """
    def __init__(self, num_controls: int, ctrl_state=None, relative_phase=False, action_only=False):
        """
        Parameters
        ----------
        num_controls
        ctrl_state
        relative_phase
        action_only
        """
        self.control_qubits = QuantumRegister(num_controls)
        self.target_qubit = QuantumRegister(1)
        self.ctrl_state = ctrl_state
        self.relative_phase = relative_phase
        self.action_only = action_only

        num_ancilla = 0
        self.ancilla_qubits = []
        if num_controls - 2 > 0:
            num_ancilla = num_controls - 2
            self.ancilla_qubits = QuantumRegister(num_ancilla)

        super().__init__('mcx_vc_dirty', num_controls + num_ancilla + 1, [], "McxDirty")

    def _define(self):
        self.definition = QuantumCircuit(self.control_qubits,
                                         self.ancilla_qubits,
                                         self.target_qubit)

        num_ctrl = len(self.control_qubits)
        num_ancilla = num_ctrl - 2
        targets = [self.target_qubit] + self.ancilla_qubits[:num_ancilla][::-1]

        self._apply_ctrl_state()

        if num_ctrl < 3:
            self.definition.mcx(
                control_qubits=self.control_qubits,
                target_qubit=self.target_qubit,
                mode="noancilla"
            )
        elif not self.relative_phase and num_ctrl == 3:
            self.definition.append(C3XGate(), [*self.control_qubits[:], self.target_qubit], [])
        else:
            for j in range(2):
                for i, _ in enumerate(self.control_qubits):  # action part
                    if i < num_ctrl - 2:
                        if targets[i] != self.target_qubit or self.relative_phase:  # gate cancelling
                            theta = np.pi / 4.

                            # cancel rightmost gates of action part with leftmost gates of reset part
                            if self.relative_phase and targets[i] == self.target_qubit and j == 1:
                                self.definition.cx(self.ancilla_qubits[num_ancilla - i - 1], targets[i])
                                self.definition.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                                self.definition.cx(self.control_qubits[num_ctrl - i - 1], targets[i])
                                self.definition.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                            else:
                                self.definition.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                                self.definition.cx(self.control_qubits[num_ctrl - i - 1], targets[i])
                                self.definition.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                                self.definition.cx(self.ancilla_qubits[num_ancilla - i - 1], targets[i])

                        else:
                            self.definition.ccx(
                                control_qubit1=self.control_qubits[num_ctrl - i - 1],
                                control_qubit2=self.ancilla_qubits[num_ancilla - i - 1],
                                target_qubit=targets[i]
                            )
                    else:
                        theta = np.pi / 4.

                        self.definition.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                        self.definition.cx(self.control_qubits[num_ctrl - i - 2], targets[i])
                        self.definition.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                        self.definition.cx(self.control_qubits[num_ctrl - i - 1], targets[i])
                        self.definition.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                        self.definition.cx(self.control_qubits[num_ctrl - i - 2], targets[i])
                        self.definition.u(theta=theta, phi=0., lam=0., qubit=targets[i])

                        break

                for i, _ in enumerate(self.ancilla_qubits[1:]):  # reset part
                    theta = np.pi / 4.

                    self.definition.cx(self.ancilla_qubits[i], self.ancilla_qubits[i + 1])
                    self.definition.u(theta=theta, phi=0., lam=0., qubit=self.ancilla_qubits[i + 1])
                    self.definition.cx(self.control_qubits[2 + i], self.ancilla_qubits[i + 1])
                    self.definition.u(theta=theta, phi=0., lam=0., qubit=self.ancilla_qubits[i + 1])

                if self.action_only:
                    self.definition.ccx(
                        control_qubit1=self.control_qubits[-1],
                        control_qubit2=self.ancilla_qubits[-1],
                        target_qubit=self.target_qubit
                    )

                    break

        self._apply_ctrl_state()

McxVchainDirty._apply_ctrl_state = _apply_ctrl_state

def mcx_v_chain_dirty(
    self,
    control_qubits,
    target_qubit,
    ancilla_qubits,
    ctrl_state=None,
    relative_phase=False,
    action_only=False
):
    """
        In-place application of k-controlled X gate with k - 2 dirty ancilla qubits.
        This decomposition uses the optimization shown in Lemma 8 of
        https://arxiv.org/abs/1501.06911), which contains at most 8k - 6 `cx`.

        Parameters
        ----------
        self : quantum circuit where the k-controlled X will be applied
        control_qubits : k control qubits
        ancilla_qubits : at least k - 2 dirty ancilla qubits
        target_qubit : target qubit of the operation
        ctrl_state
        relative_phase
        action_only
    """

    num_ctrl = len(control_qubits)
    num_ancilla = num_ctrl - 2
    targets = [target_qubit] + ancilla_qubits[:num_ancilla][::-1]

    if ctrl_state is not None:
        for i, ctrl in enumerate(ctrl_state[::-1]):
            if ctrl == '0':
                self.x(control_qubits[i])

    if num_ctrl < 3:
        self.mcx(
            control_qubits=control_qubits,
            target_qubit=target_qubit,
            mode="noancilla"
        )
    elif num_ctrl == 3:
        self.append(C3XGate(), [*control_qubits[:], target_qubit], [])
    else:
        for j in range(2):
            for i, _ in enumerate(control_qubits):      # action part
                if i < num_ctrl - 2:
                    if targets[i] != target_qubit or relative_phase:  # gate cancelling
                        theta = np.pi / 4.

                        # cancel rightmost gates of action part with leftmost gates of reset part
                        if relative_phase and targets[i] == target_qubit and j == 1:
                            self.cx(ancilla_qubits[num_ancilla - i - 1], targets[i])
                            self.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                            self.cx(control_qubits[num_ctrl - i - 1], targets[i])
                            self.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                        else:
                            self.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                            self.cx(control_qubits[num_ctrl - i - 1], targets[i])
                            self.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                            self.cx(ancilla_qubits[num_ancilla - i - 1], targets[i])

                    else:
                        self.ccx(
                            control_qubit1=control_qubits[num_ctrl - i - 1],
                            control_qubit2=ancilla_qubits[num_ancilla - i - 1],
                            target_qubit=targets[i]
                        )
                else:
                    theta = np.pi / 4.

                    self.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                    self.cx(control_qubits[num_ctrl - i - 2], targets[i])
                    self.u(theta=-theta, phi=0., lam=0., qubit=targets[i])
                    self.cx(control_qubits[num_ctrl - i - 1], targets[i])
                    self.u(theta=theta, phi=0., lam=0., qubit=targets[i])
                    self.cx(control_qubits[num_ctrl - i - 2], targets[i])
                    self.u(theta=theta, phi=0., lam=0., qubit=targets[i])

                    break

            for i, _ in enumerate(ancilla_qubits[1:]):      # reset part
                theta = np.pi / 4.

                self.cx(ancilla_qubits[i], ancilla_qubits[i + 1])
                self.u(theta=theta, phi=0., lam=0., qubit=ancilla_qubits[i + 1])
                self.cx(control_qubits[2 + i], ancilla_qubits[i + 1])
                self.u(theta=theta, phi=0., lam=0., qubit=ancilla_qubits[i + 1])

            if action_only:
                break

    if ctrl_state is not None:
        for i, ctrl in enumerate(ctrl_state[::-1]):
            if ctrl == '0':
                self.x(control_qubits[i])

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
        super().__init__('linear_mcx', num_controls + 2, [], "mcx")

    def _define(self):
        self.definition = QuantumCircuit(self.num_qubits)
        self.control_qubits = list(range(self.num_qubits - 2))
        self.target_qubit = self.num_qubits - 2,
        self.ancilla_qubit = self.num_qubits - 1

        self._apply_ctrl_state()
        if self.num_qubits < 5:
            self.definition.mcx(
                control_qubits=self.control_qubits,
                target_qubit=self.target_qubit,
                mode="noancilla"
            )
        elif self.num_qubits == 5:
            self.definition.append(C3XGate(), [*self.control_qubits[:], self.target_qubit], [])
        elif self.num_qubits == 6:
            self.definition.append(C4XGate(), [*self.control_qubits[:], self.target_qubit], [])
        elif self.num_qubits == 7:
            self.definition.append(C3XGate(), [*self.control_qubits[:3], self.ancilla_qubit], [])
            self.definition.append(C3XGate(), [*self.control_qubits[3:], self.ancilla_qubit, self.target_qubit], [])
            self.definition.append(C3XGate(), [*self.control_qubits[:3], self.ancilla_qubit], [])
            self.definition.append(C3XGate(), [*self.control_qubits[3:], self.ancilla_qubit, self.target_qubit], [])
        else:
            num_ctrl = len(self.control_qubits)

            # split controls to halve the number of qubits used for each mcx
            k_2 = int(np.ceil(self.num_qubits / 2.))
            k_1 = num_ctrl - k_2 + 1

            # when relative_phase=True only approximate Toffoli is applied because only aux qubits are targeted
            first_gate = McxVchainDirty(k_1, relative_phase=True).definition
            second_gate = McxVchainDirty(k_2).definition
            self.definition.append(first_gate,
                                   self.control_qubits[:k_1] + self.control_qubits[k_1:k_1 + k_1 - 2] + [self.ancilla_qubit])

            self.definition.append(second_gate, [*self.control_qubits[k_1:],
                                      self.ancilla_qubit] + self.control_qubits[k_1 - k_2 + 2:k_1] + [self.target_qubit])

            self.definition.append(first_gate,
                                   self.control_qubits[:k_1] + self.control_qubits[k_1:k_1 + k_1 - 2] + [self.ancilla_qubit])

            # when action_only=True only the action part of the circuit happens due to gate cancelling
            last_gate = McxVchainDirty(k_2, action_only=self.action_only).definition
            self.definition.append(last_gate,
                                   [*self.control_qubits[k_1:], self.ancilla_qubit] + self.control_qubits[k_1 - k_2 + 2:k_1] + [self.target_qubit])

        self._apply_ctrl_state()

LinearMcx._apply_ctrl_state = _apply_ctrl_state

def linear_mcx(
    self,
    control_qubits,
    target_qubit,
    ancilla_qubit,
    action_only=False
):
    """
        Linear-depth implementation of multicontrolled X with one dirty ancilla
        following the decomposition first shown in Barenco et al. 1995 (arXiv:quant-ph/9503016)
    """

    if self.num_qubits < 5:
        self.mcx(
            control_qubits=control_qubits,
            target_qubit=target_qubit,
            mode="noancilla"
        )
    elif self.num_qubits == 5:
        self.append(C3XGate(), [*control_qubits[:], target_qubit], [])
    elif self.num_qubits == 6:
        self.append(C4XGate(), [*control_qubits[:], target_qubit], [])
    elif self.num_qubits == 7:
        self.append(C3XGate(), [*control_qubits[:3], ancilla_qubit], [])
        self.append(C3XGate(), [*control_qubits[3:], ancilla_qubit, target_qubit], [])
        self.append(C3XGate(), [*control_qubits[:3], ancilla_qubit], [])
        self.append(C3XGate(), [*control_qubits[3:], ancilla_qubit, target_qubit], [])
    else:
        num_ctrl = len(control_qubits)

        # split controls to halve the number of qubits used for each mcx
        k_1 = int(np.ceil((num_ctrl + 1.) / 2.))
        k_2 = int(np.floor((num_ctrl + 1.) / 2.))


        first_gate = McxVchainDirty(k_1).definition
        second_gate = McxVchainDirty(k_2).definition
        self.append(first_gate,
                    control_qubits[:k_1] + control_qubits[k_1:k_1 + k_1 - 2] + [ancilla_qubit])

        self.append(second_gate, [*control_qubits[k_1:],
                                  ancilla_qubit] + control_qubits[k_1 - k_2 + 2:k_1] + [target_qubit])

        self.append(first_gate,
                    control_qubits[:k_1] + control_qubits[k_1:k_1 + k_1 - 2] + [ancilla_qubit])

        last_gate = McxVchainDirty(k_2, action_only=action_only).definition
        self.append(last_gate, [*control_qubits[k_1:],
                                ancilla_qubit] + control_qubits[k_1 - k_2 + 2:k_1] + [target_qubit])

