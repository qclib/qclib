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

""" Test linear mcx with ancilla """

from unittest import TestCase

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator
from qclib.gates.mcx import McxVchainDirty, LinearMcx


def apply_control_state_on_quantum_circuit(
    quantum_circuit: QuantumCircuit, control_qubits: QuantumRegister, ctrl_state: str
):
    """
    Applies the X gate to the corresponding qubit in which the
    bit string describes it to be in state 0 as a means to simulate
    open controlled operations on a specific set of qubits. This operations
    are applied using Qiskit's QuantumCircuit object instead of Gate object.

    Parameters
    ----------
        quantum_circuit : QuantumCircuit object on which the X gates are to be applied
        control_bits    : QuantumRegister containing with the qubits to be used as control
        ctrl_state  : String of binary digits describing wich state is used as control
                    in the multicontrolled operation
    """
    if ctrl_state is not None:
        for i, ctrl in enumerate(ctrl_state[::-1]):
            if ctrl == "0":
                quantum_circuit.x(control_qubits[i])


class TestLinearMCX(TestCase):
    """Testing qclib.gates.mcx_gate"""

    def test_linear_mcx(self):
        """Test if linear_mcx is correct"""
        for num_qubits in range(8, 9):
            self._operator_cmp(
                num_qubits=num_qubits, mcx_method=LinearMcx, mode="recursion"
            )

    def _compare_linear_mcx_action_only(self, num_qubits, ctrl_state=None):
        """Test if linear_mcx is correct"""

        linear_circuit = QuantumCircuit(num_qubits)
        qiskit_circuit = QuantumCircuit(num_qubits)
        theta = np.random.uniform(0.0, 2.0 * np.pi)

        mcx_method = LinearMcx(
            num_qubits - 2, ctrl_state=ctrl_state, action_only=True
        ).definition

        linear_circuit.append(mcx_method, list(range(num_qubits)))
        linear_circuit.rz(theta, num_qubits - 1)
        linear_circuit.append(mcx_method.inverse(), list(range(num_qubits)))

        mcx_qiskit = self._build_qiskit_method_mcx_recursive(
            num_qubits=num_qubits, ctrl_state=ctrl_state, mode="recursion"
        )

        qiskit_circuit.append(mcx_qiskit, list(range(num_qubits)))
        qiskit_circuit.rz(theta, num_qubits - 1)
        qiskit_circuit.append(mcx_qiskit, list(range(num_qubits)))

        linear_op = Operator(linear_circuit).data
        qiskit_op = Operator(qiskit_circuit).data

        self.assertTrue(np.allclose(linear_op, qiskit_op))

    def test_linear_mcx_action_only(self):
        """Test linear mcx action only"""
        for num_qubits in range(8, 9):
            self._compare_linear_mcx_action_only(num_qubits)

    def test_linear_mcx_action_only_random_ctrl_state(self):
        """Test if linear_mcx is correct"""
        num_qubit_range = list(range(8, 9))
        basis_states = [
            f"{np.random.randint(2 ** (n_ctrl - 2)):0{n_ctrl - 2}b}"
            for n_ctrl in num_qubit_range
        ]

        for num_qubits, ctrl_state in zip(num_qubit_range, basis_states):
            self._compare_linear_mcx_action_only(num_qubits, ctrl_state)

    def test_linear_mcx_depth(self):
        """Test linear_mcx depth"""

        for num_qubits in range(10, 11):
            mcx_dirty_ancilla = LinearMcx(num_qubits - 2).definition

            mcx_qiskit = QuantumCircuit(num_qubits)

            mcx_qiskit.mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1,
                mode="recursion",
            )

            tr_mcx_dirty_ancilla = transpile(mcx_dirty_ancilla, basis_gates=["u", "cx"])
            tr_mcx_qiskit = transpile(mcx_qiskit, basis_gates=["u", "cx"])

            self.assertLess(tr_mcx_dirty_ancilla.depth(), tr_mcx_qiskit.depth())

    def _operator_cmp(
        self,
        num_qubits,
        mcx_method: LinearMcx,
        mode: str,
        ctrl_state: str = None,
        action_only=False,
    ):
        """
        Compares if the custom operator defined by the custom MCX method is the same
        as the one defined by Qiskit MCX method.
        Parameters
        ----------
        params:
            qubit_range : The number of qubits with which our method needs to be tested
            McxMethod   : The class definition of the method to be used. It must be `LinearMcx`

            action_only : Decide wether or not use only the action of the V-Chain of Toffoli
                        gates
        """
        # for num_qubits in qubit_range:
        mcx_method = mcx_method(
            num_qubits - 2, ctrl_state=ctrl_state, action_only=action_only
        ).definition

        mcx_qiskit = self._build_qiskit_method_mcx_recursive(
            num_qubits=num_qubits, ctrl_state=ctrl_state, mode=mode
        )

        mcx_method_op = Operator(mcx_method).data
        mcx_qiskit_op = Operator(mcx_qiskit).data

        self.assertTrue(np.allclose(mcx_method_op, mcx_qiskit_op))

    def _build_qiskit_method_mcx_recursive(
        self,
        num_qubits,
        ctrl_state: str = None,
        mode="recursive",
    ):
        """
        Bulds qiskit quantum circuit with mcx-recursive
        method to be used as reference for comparison

        Parameters
        ----------
            num_qubits : Total number of qubits on the system
            ctrl_state : string with binary digits that specifies the control state
            mode : Decomposition mode to be used for multicontrolled operation
        """
        num_controls = num_qubits - 2
        control_qubits = QuantumRegister(num_controls)
        target_qubit = QuantumRegister(1)
        ancilla_qubits = QuantumRegister(1)

        mcx_qiskit = QuantumCircuit(
            control_qubits,
            target_qubit,
            ancilla_qubits,
        )

        apply_control_state_on_quantum_circuit(
            quantum_circuit=mcx_qiskit,
            control_qubits=control_qubits,
            ctrl_state=ctrl_state,
        )

        mcx_qiskit.mcx(
            control_qubits=control_qubits,
            target_qubit=target_qubit,
            ancilla_qubits=ancilla_qubits,
            mode=mode,
        )

        apply_control_state_on_quantum_circuit(
            quantum_circuit=mcx_qiskit,
            control_qubits=control_qubits,
            ctrl_state=ctrl_state,
        )

        return mcx_qiskit


class TestMcxVchainDirty(TestCase):
    """Test class McxVchainDirty"""

    def test_mcx_v_chain_dirty_depth(self):
        """Test mcx_v_chain_dirty depth"""

        for num_controls in range(10, 11):
            num_ancilla = num_controls - 2
            control_qubits = QuantumRegister(num_controls)
            ancilla_qubits = QuantumRegister(num_ancilla)
            target_qubit = QuantumRegister(1)

            mcx_v_chain = McxVchainDirty(num_controls).definition

            mcx_v_chain_qiskit = QuantumCircuit(
                control_qubits, ancilla_qubits, target_qubit
            )

            mcx_v_chain_qiskit.mcx(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits,
                mode="v-chain-dirty",
            )

            tr_mcx_v_chain = transpile(mcx_v_chain, basis_gates=["u", "cx"])
            tr_mcx_v_chain_qiskit = transpile(
                mcx_v_chain_qiskit, basis_gates=["u", "cx"]
            )

            self.assertTrue(8 * num_controls - 6 == tr_mcx_v_chain.count_ops()["cx"])
            self.assertLess(tr_mcx_v_chain.depth(), tr_mcx_v_chain_qiskit.depth())

    def test_mcx_v_chain_dirty(self):
        """Test McxVchainDirty"""
        for num_controls in range(6, 7):
            self._operator_cmp(
                num_controls=num_controls,
                mcx_method=McxVchainDirty,
                mode="v-chain-dirty",
            )

    def test_mcx_v_chain_dirty_random_ctrl_state(self):
        """
        Test if mcx_v_chain_dirty is correct
        with non trivial randomly generated
        control states
        """
        control_qubit_range = list(range(6, 7))
        basis_states = [
            f"{np.random.randint(2 ** n_ctrl):0{n_ctrl}b}"
            for n_ctrl in control_qubit_range
        ]

        for num_controls, ctrl_state in zip(control_qubit_range, basis_states):
            self._operator_cmp(
                num_controls=num_controls,
                mcx_method=McxVchainDirty,
                mode="v-chain-dirty",
                ctrl_state=ctrl_state,
            )
    
    def test_mcx_v_chain_3targets(self):
        """Test multiple targets McxVchainDirty"""

        # mcx_v_chain_circuit
        num_controls = 4
        num_target_qubit = 3
        mcx_v_chain_circuit = McxVchainDirty(
            num_controls, num_target_qubit=num_target_qubit
        ).definition

        # qiskit_circuit
        qiskit_circuit = QuantumCircuit(9)
        controls_idx = list(np.arange(4))
        for target_idx in range(6, 9):
            qiskit_circuit.mcx(controls_idx, [target_idx])

        mcx_v_chain_op = Operator(mcx_v_chain_circuit).data
        qiskit_mcx_op = Operator(qiskit_circuit).data

        tr_mcx_v_chain = transpile(mcx_v_chain_circuit, basis_gates=["u", "cx"])

        np.allclose(mcx_v_chain_op, qiskit_mcx_op)

        if num_controls > 3:
            self.assertTrue(
                10 + (num_controls - 2) * 8 + (num_target_qubit - 1) * 12
                == tr_mcx_v_chain.count_ops()["cx"]
            )


    def _operator_cmp(
        self,
        num_controls,
        mcx_method: McxVchainDirty,
        mode: str,
        ctrl_state: str = None,
        action_only=False,
    ):
        """
        Compares if the custom operator defined by the custom MCX method is the same
        as the one defined by Qiskit MCX method.

        Parameters
        ----------
            control_qubit_range: The number of control qubits
            mcx_method: The class definition of the method to be used. It must be `McxVchainDirty`

            action_only: Decide wether or not use only the action of the V-Chain of Toffoli
                        gates
        """
        mcx_method = mcx_method(
            num_controls,
            ctrl_state=ctrl_state,
            action_only=action_only
        ).definition

        # defining quiskit's
        mcx_v_chain_qiskit = self._build_qiskit_method_mcx_vchain_dirty(
            num_controls=num_controls, mode=mode, ctrl_state=ctrl_state
        )

        mcx_method_op = Operator(mcx_method).data
        mcx_v_chain_qiskit_op = Operator(mcx_v_chain_qiskit).data

        self.assertTrue(np.allclose(mcx_method_op, mcx_v_chain_qiskit_op))

    def _build_qiskit_method_mcx_vchain_dirty(
        self,
        num_controls,
        mode: str,
        ctrl_state: str = None,
    ):
        """
        Bulds qiskit quantum circuit with mcx-vchain-dirty
        method to be used as reference for comparison

        Parameters
        ----------
            num_controls    : Total number of control qubits on the system
            ctrl_state  : string with binary digits that specifies the control state
            mode    : Decomposition mode to be used for multicontrolled operation
        """
        num_ancilla = num_controls - 2
        control_qubits = QuantumRegister(num_controls)
        ancilla_qubits = QuantumRegister(num_ancilla)
        target_qubit = QuantumRegister(1)

        mcx_qiskit = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

        apply_control_state_on_quantum_circuit(
            quantum_circuit=mcx_qiskit,
            control_qubits=control_qubits,
            ctrl_state=ctrl_state,
        )

        mcx_qiskit.mcx(
            control_qubits=control_qubits,
            target_qubit=target_qubit,
            ancilla_qubits=ancilla_qubits,
            mode=mode,
        )

        apply_control_state_on_quantum_circuit(
            quantum_circuit=mcx_qiskit,
            control_qubits=control_qubits,
            ctrl_state=ctrl_state,
        )

        return mcx_qiskit
