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
from qclib.gates.mcx_gate import mcx_v_chain_dirty
from qclib.gates.mcx_gate import linear_mcx


class TestLinearMCX(TestCase):
    """ Testing qclib.gates.mcx_gate """

    QuantumCircuit.mcx_v_chain_dirty = mcx_v_chain_dirty
    QuantumCircuit.linear_mcx = linear_mcx


    def test_mcx_v_chain_dirty(self):
        """ Test if mcx_v_chain_dirty is correct """

        for num_controls in range(6, 8):
            num_ancilla = num_controls - 2
            control_qubits = QuantumRegister(num_controls)
            ancilla_qubits = QuantumRegister(num_ancilla)
            target_qubit = QuantumRegister(1)

            mcx_v_chain = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain.mcx_v_chain_dirty(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits
            )

            mcx_v_chain_qiskit = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain_qiskit.mcx(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits,
                mode="v-chain-dirty"
            )

            mcx_v_chain_op = Operator(mcx_v_chain).data
            mcx_v_chain_qiskit_op = Operator(mcx_v_chain_qiskit).data

            self.assertTrue(np.allclose(mcx_v_chain_op, mcx_v_chain_qiskit_op))


    def test_mcx_v_chain_dirty_depth(self):
        """ Test mcx_v_chain_dirty depth"""

        for num_controls in range(30, 31):
            num_ancilla = num_controls - 2
            control_qubits = QuantumRegister(num_controls)
            ancilla_qubits = QuantumRegister(num_ancilla)
            target_qubit = QuantumRegister(1)

            mcx_v_chain = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain.mcx_v_chain_dirty(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits
            )

            mcx_v_chain_qiskit = QuantumCircuit(control_qubits, ancilla_qubits, target_qubit)

            mcx_v_chain_qiskit.mcx(
                control_qubits=control_qubits,
                target_qubit=target_qubit,
                ancilla_qubits=ancilla_qubits,
                mode="v-chain-dirty"
            )

            tr_mcx_v_chain = transpile(mcx_v_chain, basis_gates=['u', 'cx'])
            tr_mcx_v_chain_qiskit = transpile(mcx_v_chain_qiskit, basis_gates=['u', 'cx'])

            self.assertLess(tr_mcx_v_chain.depth(), tr_mcx_v_chain_qiskit.depth())


    def test_linear_mcx(self):
        """ Test if linear_mcx is correct """

        for num_qubits in range(6, 8):
            mcx_dirty_ancilla = QuantumCircuit(num_qubits)

            mcx_dirty_ancilla.linear_mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1
            )

            mcx_qiskit = QuantumCircuit(num_qubits)

            mcx_qiskit.mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1,
                mode="recursion"
            )

            mcx_dirty_ancilla_op = Operator(mcx_dirty_ancilla).data
            mcx_qiskit_op = Operator(mcx_qiskit).data

            self.assertTrue(np.allclose(mcx_dirty_ancilla_op, mcx_qiskit_op))


    def test_linear_mcx_depth(self):
        """ Test linear_mcx depth"""

        for num_qubits in range(30, 31):
            mcx_dirty_ancilla = QuantumCircuit(num_qubits)

            mcx_dirty_ancilla.linear_mcx(
                control_qubits=list(range(num_qubits - 2)),
                target_qubit=num_qubits - 2,
                ancilla_qubits=num_qubits - 1
            )

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
