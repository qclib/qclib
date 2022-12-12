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
from qiskit import QuantumCircuit
from qiskit.circuit.library import C3XGate, C4XGate


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
        circuit : quantum circuit where the k-controlled X will be applied
        control_qubits : k control qubits
        ancilla_qubits : at least k - 2 dirty ancilla qubits
        target_qubit : target qubit of the operation
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
                        if (relative_phase and targets[i] == target_qubit and j == 1):
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


QuantumCircuit.mcx_v_chain_dirty = mcx_v_chain_dirty


def linear_mcx(
    self,
    control_qubits,
    target_qubit,
    ancilla_qubits
):
    """
        Linear-depth implementation of multicontrolled X with one dirty ancilla
        following the decomposition first shown in
            https://link.aps.org/doi/10.1103/PhysRevA.52.3457
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
        self.append(C3XGate(), [*control_qubits[:3], ancilla_qubits], [])
        self.append(C3XGate(), [*control_qubits[3:], ancilla_qubits, target_qubit], [])
        self.append(C3XGate(), [*control_qubits[:3], ancilla_qubits], [])
        self.append(C3XGate(), [*control_qubits[3:], ancilla_qubits, target_qubit], [])
    else:
        num_ctrl = len(control_qubits)

        # split controls to halve the number of qubits used for each mcx
        k_1 = int(np.ceil((num_ctrl + 1.) / 2.))
        k_2 = int(np.floor((num_ctrl + 1.) / 2.))

        self.mcx_v_chain_dirty(
            control_qubits=control_qubits[:k_1],
            target_qubit=ancilla_qubits,
            ancilla_qubits=control_qubits[k_1:k_1 + k_1 - 2]
        )
        self.mcx_v_chain_dirty(
            control_qubits=[*control_qubits[k_1:], ancilla_qubits],
            target_qubit=target_qubit,
            ancilla_qubits=control_qubits[k_1 - k_2 + 2:k_1]
        )
        self.mcx_v_chain_dirty(
            control_qubits=control_qubits[:k_1],
            target_qubit=ancilla_qubits,
            ancilla_qubits=control_qubits[k_1:k_1 + k_1 - 2]
        )
        self.mcx_v_chain_dirty(
            control_qubits=[*control_qubits[k_1:], ancilla_qubits],
            target_qubit=target_qubit,
            ancilla_qubits=control_qubits[k_1 - k_2 + 2:k_1]
        )


QuantumCircuit.linear_mcx = linear_mcx
