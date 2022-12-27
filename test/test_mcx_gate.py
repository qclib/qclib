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
from typing import Union
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator
from qclib.gates.mcx_gate import mcx_v_chain_dirty
from qclib.gates.mcx_gate import linear_mcx, McxVchainDirty, LinearMcx

class TestLinearMCX(TestCase):
    """ Testing qclib.gates.mcx_gate """

    #QuantumCircuit.mcx_v_chain_dirty = mcx_v_chain_dirty
    QuantumCircuit.linear_mcx = linear_mcx

    def test_linear_mcx(self):
        """ Test if linear_mcx is correct """
        self._operator_cmp_loop(
            qubit_range=range(8, 9),
            McxMethod=LinearMcx,
            mode="recursion"
        )

    def test_linear_mcx_action_only(self):
        """ Test if linear_mcx is correct """
        self._operator_cmp_loop(
            qubit_range=range(8, 9),
            McxMethod=LinearMcx,
            mode="recursion",
            action_only=True
        )

    def test_linear_mcx_depth(self):
        """ Test linear_mcx depth"""

        for num_qubits in range(30, 31):

            mcx_dirty_ancilla = LinearMcx(num_qubits-2).definition

            mcx_qiskit = QuantumCircuit(num_qubits)

            mcx_qiskit.mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1,
                mode="recursion"
            )

            tr_mcx_dirty_ancilla = transpile(mcx_dirty_ancilla, basis_gates=['u', 'cx'])
            tr_mcx_qiskit = transpile(mcx_qiskit, basis_gates=['u', 'cx'])

            self.assertLess(tr_mcx_dirty_ancilla.depth(), tr_mcx_qiskit.depth())

    def _operator_cmp_loop(
        self,
        qubit_range,
        McxMethod: LinearMcx,
        mode: str,
        action_only=False
    ):
        """
        Compares if the custom operator defined by the custom MCX method is the same
        as the one defined by Qiskit MCX method.
        Parameters
        ----------
        params: 
            qubit_range: The number of qubits with which our method needs to be tested
            McxMethod: The class definition of the method to be used. It must be `LinearMcx`

            action_only: Decide wether or not use only the action of the V-Chain of Toffoli
                        gates
        """
        for num_qubits in qubit_range:
            mcx_method = McxMethod(num_qubits-2, action_only=action_only).definition

            mcx_qiskit = QuantumCircuit(num_qubits)

            mcx_qiskit.mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1,
                mode=mode
            )

            mcx_method_op = Operator(mcx_method).data
            mcx_qiskit_op = Operator(mcx_qiskit).data

            self.assertTrue(np.allclose(mcx_method_op, mcx_qiskit_op))


class TestMcxVchainDirty(TestCase):

    QuantumCircuit.mcx_v_chain_dirty = mcx_v_chain_dirty

    def test_mcx_v_chain_dirty_depth(self):
        """ Test mcx_v_chain_dirty depth"""

        for num_controls in range(30, 31):
            num_ancilla = num_controls - 2
            control_qubits = QuantumRegister(num_controls)
            ancilla_qubits = QuantumRegister(num_ancilla)
            target_qubit = QuantumRegister(1)

            mcx_v_chain = McxVchainDirty(num_controls).definition

            mcx_v_chain_qiskit = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain_qiskit.mcx(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits,
                mode="v-chain-dirty"
            )

            tr_mcx_v_chain = transpile(mcx_v_chain, basis_gates=['u', 'cx'])
            tr_mcx_v_chain_qiskit = transpile(mcx_v_chain_qiskit, basis_gates=['u', 'cx'])
            
            self.assertTrue(8 * num_controls - 6 == tr_mcx_v_chain.count_ops()['cx'])
            self.assertLess(tr_mcx_v_chain.depth(), tr_mcx_v_chain_qiskit.depth())

    def test_mcx_v_chain_dirty(self):
        """ Test if mcx_v_chain_dirty is correct """
        self._operator_cmp_loop(
            control_qubit_range=range(4, 6),
            McxMethod=McxVchainDirty,
            mode="v-chain-dirty"
        )

    def test_mcx_v_chain_dirty_action_only(self):
        """ Test if mcx_v_chain_dirty is correct with action only"""
        self._operator_cmp_loop(
            control_qubit_range=range(4, 6),
            McxMethod=McxVchainDirty,
            mode="v-chain-dirty",
            action_only=True
        )

    def _operator_cmp_loop(
        self,
        control_qubit_range,
        McxMethod: McxVchainDirty,
        mode: str,
        action_only=False
    ):
        """
        Compares if the custom operator defined by the custom MCX method is the same
        as the one defined by Qiskit MCX method.
        Parameters
        ----------
        params: 
            control_qubit_range: The number of control qubits with which our method must to be tested
            McxMethod: The class definition of the method to be used. It must be `LinearMcx`

            action_only: Decide wether or not use only the action of the V-Chain of Toffoli
                        gates
        """
        for num_controls in control_qubit_range:
            mcx_method = McxMethod(num_controls, action_only=action_only).definition

            #defining quiskit's 
            num_ancilla = num_controls - 2
            control_qubits = QuantumRegister(num_controls)
            ancilla_qubits = QuantumRegister(num_ancilla)
            target_qubit = QuantumRegister(1)

            mcx_v_chain_qiskit = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain_qiskit.mcx(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits,
                mode=mode
            )

            mcx_method_op = Operator(mcx_method).data
            mcx_v_chain_qiskit_op = Operator(mcx_v_chain_qiskit).data

            self.assertTrue(np.allclose(mcx_method_op, mcx_v_chain_qiskit_op))
