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

from __future__ import annotations
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from qiskit.circuit.library import C3XGate, C4XGate
from qiskit.circuit import Gate
from qclib.gates.toffoli import Toffoli
from qclib.gates.util import apply_ctrl_state

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

    @staticmethod
    def toffoli_multi_target(num_targets, side=None):
        """ " """

        size = 2 + num_targets
        circuit = QuantumCircuit(size)
        if side == "l":
            for i in range(num_targets - 1):
                circuit.name = 'toff_left'
                circuit.cx(size - i - 2, size - i - 1)
            circuit.ccx(0, 1, 2)
            return circuit

        elif side == "r":
            circuit.name = 'toff_right'
            circuit.ccx(0, 1, 2)
            for i in range(num_targets - 1):
                circuit.cx(i + 2, i + 3)
            return circuit

        elif side is None:
            circuit.name = 'toff_l_and_r'
            for i in range(num_targets - 1):
                circuit.cx(size - i - 2, size - i - 1)
            circuit.ccx(0, 1, 2)
            for i in range(num_targets - 1):
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

        if num_ctrl == 2:
            self.definition.append(
                self.toffoli_multi_target(len(self.target_qubits)),
                [*self.control_qubits, *self.target_qubits],
            )
        elif num_ctrl == 1:
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
            side = 'l'
            for j in range(2):
                self._action_circuit(j, num_ancilla, num_ctrl, targets_aux, side)
                side = 'r'
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
                        self.toffoli_multi_target(num_targets, side),
                        [control_1, control_2, *targets],
                    )

                    break

        self._apply_ctrl_state()

    def _action_circuit(self, j, num_ancilla, num_ctrl, targets_aux, side):
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
                        self.toffoli_multi_target(num_targets, side),
                        [control_1, control_2, *targets],
                    )

            else:
                controls = [
                    self.control_qubits[num_ctrl - i - 2],
                    self.control_qubits[num_ctrl - i - 1],
                ]

                self.definition.append(Toffoli(), [*controls, targets_aux[i]])

                break

    @staticmethod
    def mcx_vchain_dirty(
        circuit,
        controls=None,
        target=None,
        ctrl_state=None,
        relative_phase=False,
        action_only=False,
    ):
        """
        Implementation based on lemma 8 of Iten et al. (2016) arXiv:1501.06911.
        Decomposition of a multicontrolled X gate with at least k <= ceil(n/2) ancilae
        for n as the total number of qubits in the system. It also includes optimizations
        using approximated Toffoli gates up to a diagonal.
        """
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
                + self.control_qubits[k_1 : k_1 + k_1 - 2]
                + [self.ancilla_qubit],
            )

            self.definition.append(
                second_gate,
                [*self.control_qubits[k_1:], self.ancilla_qubit]
                + self.control_qubits[k_1 - k_2 + 2 : k_1]
                + [self.target_qubit],
            )

            self.definition.append(
                first_gate,
                self.control_qubits[:k_1]
                + self.control_qubits[k_1 : k_1 + k_1 - 2]
                + [self.ancilla_qubit],
            )

            last_gate = McxVchainDirty(k_2, action_only=self.action_only).definition
            self.definition.append(
                last_gate,
                [*self.control_qubits[k_1:], self.ancilla_qubit]
                + self.control_qubits[k_1 - k_2 + 2 : k_1]
                + [self.target_qubit],
            )

        self._apply_ctrl_state()

    @staticmethod
    def mcx(circuit, controls=None, target=None, ctrl_state=None, action_only=False):
        """
        Implementation based on lemma 9 of Iten et al. (2016) arXiv:1501.06911.
        Decomposition of a multicontrolled X gate with a dirty ancilla by splitting
        it into two sequences of two alternating multicontrolled X gates on
        k1 = ceil((n+1)/2) and k2 = floor((n+1)/2) qubits. For n the total
        number of qubits in the system. Where it also reuses some optimizations available
        """
        circuit.append(
            LinearMcx(len(controls), ctrl_state, action_only), [*controls, target]
        )


LinearMcx._apply_ctrl_state = apply_ctrl_state


def get_linear_depth_ladder_ops(qreg: list[int]) -> tuple[QuantumCircuit, list[int]]:
    """
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Args:
        List of qubit indices to apply the ladder operations on. qreg[0] is assumed to be ancilla.

    Returns:
        A tuple consisting of the linear-depth ladder circuit and the index of control qubit to
        apply the final CCX gate.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    n = len(qreg)
    if n <= 3:
        raise ValueError("n = n_ctrls + 1 => n_ctrls >= 3 to use MCX ladder. Otherwise, use CCX")

    qc = QuantumCircuit(n)

    # up-ladder
    for i in range(2, n - 2, 2):
        qc.compose(
            CCXN(1).definition,
            qubits=[qreg[i + 1],
                    qreg[i + 2], qreg[i]],
            inplace=True
        )

    # down-ladder
    if n % 2 != 0:
        x, y, t = n - 3, n - 5, n - 6
    else:
        x, y, t = n - 1, n - 4, n - 5

    if t > 0:
        qc.compose(
            CCXN(1).definition.inverse(),
            qubits=[qreg[x], qreg[y], qreg[t]],
            inplace=True
        )

    for i in range(t, 2, -2):
        qc.compose(
            CCXN(1).definition.inverse(),
            qubits=[qreg[i], qreg[i - 1], qreg[i - 2]],
            inplace=True
        )

    mid_second_ctrl = 1 + max(0, 6 - n)
    final_ctrl = qreg[mid_second_ctrl] - 1
    return qc, final_ctrl


class MCXGidneyLinearDepth(Gate):
    """
    Synthesise a multi-controlled X gate with k controls using 1 ancillary qubit producing a circuit
    with 2k-3 Toffoli gates and O(k) depth if ancilla is clean and 4k-8 Toffoli gates and O(k) depth
    if ancilla is dirty as shown in Sec 5.1 of [1].

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def __init__(self, num_controls, clean=True):
        self.n_ctrls = num_controls
        self.n_anc = 1
        self.n_qubits = num_controls + self.n_anc + 1  # control + ancilla + target
        self.clean = clean
        super().__init__(f"linear_mcx_{num_controls}_{self.n_anc}", self.n_qubits, [])

    def _define(self):
        ctrl, targ = QuantumRegister(self.n_ctrls, "ctrl"), QuantumRegister(1, "targ")
        qc = QuantumCircuit(ctrl, targ)

        if self.n_ctrls <= 2:
            qc.mcx(ctrl, targ)
        else:
            anc = QuantumRegister(self.n_anc, "anc")
            qc.add_register(anc)
            ladder_ops, final_ctrl = get_linear_depth_ladder_ops(list(range(self.n_ctrls + self.n_anc)))
            relative_phase = CCXN(1, apply_x=False).definition
            qc.compose(relative_phase, [ctrl[0]] + [ctrl[1]] + anc[:], inplace=True) # create conditionally clean ancilla
            qc.compose(ladder_ops, anc[:] + ctrl[:], inplace=True)  #             # up-ladder
            qc.ccx(anc, ctrl[final_ctrl], targ)  #                                # target
            qc.compose(ladder_ops.inverse(), anc[:] + ctrl[:], inplace=True)  #   # down-ladder
            qc.compose(relative_phase.inverse(), [ctrl[0]] + [ctrl[1]] + anc[:], inplace=True)

            if not self.clean:
                # toggle-detection if dirty ancilla
                qc.compose(ladder_ops, anc[:] + ctrl[:], inplace=True)
                qc.ccx(anc, ctrl[final_ctrl], targ)
                qc.compose(ladder_ops.inverse(), anc[:] + ctrl[:], inplace=True)

        self.definition = qc


class CCXN(Gate):
    """
    Construct a quantum circuit for creating n-condionally clean ancillae using 3n qubits. This
    implements Fig. 4a of [1]. The order of returned qubits is x, y, target.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def __init__(self, n, apply_x=True):
        self.n = n
        self.n_qubits = 3 * n
        self.apply_x = apply_x
        super().__init__(f"ccx_{n}", self.n_qubits, [])

    def _define(self):
        q_r_ctrl = QuantumRegister(2 * self.n, name='ctrl ')
        q_r_targ = QuantumRegister(self.n, name='targ ')

        x, y, target = q_r_ctrl[:self.n], q_r_ctrl[self.n:2 * self.n], q_r_targ[:]

        qc = QuantumCircuit(q_r_ctrl, q_r_targ, name=f"ccx_{self.n}")

        if self.apply_x:
            qc.x(target)

        qc.h(target)
        qc.t(target)
        qc.cx(x, target)
        qc.tdg(target)
        qc.cx(y, target)
        qc.t(target)
        qc.cx(x, target)
        qc.tdg(target)
        qc.h(target)

        self.definition = qc



def build_logn_depth_ccx_ladder(
    ancilla_idx: int, ctrls: list[int], skip_cond_clean=False
) -> tuple[QuantumCircuit, list[int]]:
    """
    Helper function to build a log-depth ladder compose of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        ancilla_idx: Index of the ancillary qubit.
        ctrls: List of control qubits.
        skip_cond_clean: If True, do not include the conditionally clean ancilla (step 1 and 5 in
            Fig. 4b of [1]).

    Returns:
        A tuple consisting of the log-depth ladder circuit of cond. clean ancillae and the list of
        indices of control qubit to apply the linear-depth MCX gate.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    qc = QuantumCircuit(len(ctrls) + 1)
    anc = [ancilla_idx]
    final_ctrls = []

    while len(ctrls) > 1:
        next_batch_len = min(len(anc) + 1, len(ctrls))
        ctrls, nxt_batch = ctrls[next_batch_len:], ctrls[:next_batch_len]
        new_anc = []
        while len(nxt_batch) > 1:
            ccx_n = len(nxt_batch) // 2
            st = int(len(nxt_batch) % 2)
            ccx_x, ccx_y, ccx_t = (
                nxt_batch[st : st + ccx_n],
                nxt_batch[st + ccx_n :],
                anc[-ccx_n:],
            )
            assert len(ccx_x) == len(ccx_y) == len(ccx_t) == ccx_n >= 1
            if ccx_t != [ancilla_idx]:
                qc.compose(CCXN(ccx_n).definition, ccx_x + ccx_y + ccx_t, inplace=True)
            else:
                if not skip_cond_clean:
                    relative_phase = CCXN(1, apply_x=False).definition
                    qc.compose(relative_phase, ccx_x + ccx_y + ccx_t, inplace=True) # create conditionally clean ancilla
                    # qc.ccx(ccx_x[0], ccx_y[0], ccx_t[0])  #   # create conditionally clean ancilla
            new_anc += nxt_batch[st:]  #                      # newly created conditionally clean ancilla
            nxt_batch = ccx_t + nxt_batch[:st]
            anc = anc[:-ccx_n]

        anc = sorted(anc + new_anc)
        final_ctrls += nxt_batch

    final_ctrls += ctrls
    final_ctrls = sorted(final_ctrls)
    return qc, final_ctrls[:-1]  #                            # exclude ancilla


class MCXGidneyLogDepth(Gate):
    """
    Synthesise a multi-controlled X gate with k controls using 2 ancillary qubits producing a circuit
    with 2k-3 Toffoli gates and O(log(k)) depth if ancillae are clean and 4k-8 Toffoli gates and O(log(k))
    depth if ancillae are dirty as shown in  Sec. 5.3 of [1].

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    def __init__(self, num_controls, clean=True):
        self.n_ctrl = num_controls
        self.n_anc = 2
        self.num_qubits = num_controls + self.n_anc + 1  #                   # control + ancilla + target
        self.clean = clean
        super().__init__(f"log_mcx_{num_controls}_{self.n_anc}", self.num_qubits, [])

    def _define(self):
        ctrl, targ = QuantumRegister(self.n_ctrl, "ctrl"), QuantumRegister(1, "targ")
        qc = QuantumCircuit(ctrl, targ)

        if self.n_ctrl <= 2:
            qc.mcx(ctrl, targ)
        else:
            anc = QuantumRegister(self.n_anc, "anc")
            qc.add_register(anc)
            ladder_ops, final_ctrls = build_logn_depth_ccx_ladder(self.n_ctrl, list(range(self.n_ctrl)))
            qc.compose(ladder_ops, ctrl[:] + [anc[0]], inplace=True)
            if len(final_ctrls) == 1:  #                                     # Already a toffoli
                qc.ccx(anc[0], ctrl[final_ctrls[0]], targ)
            else:
                mid_mcx = MCXGidneyLinearDepth(len(final_ctrls) + 1, clean=True)
                qc.compose(
                    mid_mcx.definition,
                    [anc[0]] + ctrl[final_ctrls] + targ[:] + [anc[1]], #     # ctrls, targ, anc
                    inplace=True,
                )
            qc.compose(ladder_ops.inverse(), ctrl[:] + [anc[0]], inplace=True)

            if not self.clean:
                # toggle-detection if not clean
                ladder_ops_new, final_ctrls = build_logn_depth_ccx_ladder(
                    self.n_ctrl, list(range(self.n_ctrl)), skip_cond_clean=True
                )
                qc.compose(ladder_ops_new, ctrl[:] + [anc[0]], inplace=True)
                if len(final_ctrls) == 1:
                    qc.ccx(anc[0], ctrl[final_ctrls[0]], targ)
                else:
                    qc.compose(
                        mid_mcx.definition,
                        [anc[0]] + ctrl[final_ctrls] + targ[:] + [anc[1]],
                        inplace=True,
                    )
                qc.compose(ladder_ops_new.inverse(), ctrl[:] + [anc[0]], inplace=True)

        self.definition = qc
