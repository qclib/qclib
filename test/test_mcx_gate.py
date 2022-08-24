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

""" Test qclib.gates.mcx_gate """

import unittest
import numpy as np
import qiskit

from qiskit.quantum_info import Operator
from qclib.gates.mcx_gate import mcx
from qclib.gates.mcx_gate import mcx_no_ancilla


class TestMCXGate(unittest.TestCase):
    """
        Tests for multicontrolled NOT gate with and without ancilla
    """

    def test_mcx_no_ancilla(self):
        """
            Comparison of MCX decomposition using free qubits with Qiskit MCX
        """

        n_controls = 6
        qr_control = qiskit.QuantumRegister(n_controls, 'control')
        qr_free = qiskit.QuantumRegister(n_controls - 2, 'free')
        qr_target = qiskit.QuantumRegister(1, 'target')

        # implementation of Barenco et al. (1995) MCX with free qubits
        qc_mcx_no_ancilla = qiskit.QuantumCircuit(qr_control, qr_free, qr_target)

        mcx_no_ancilla(circuit=qc_mcx_no_ancilla, controls=qr_control, free=qr_free, targ=qr_target)

        # Qiskit implementation of MCX including unused free qubits
        qc_mcx_qiskit = qiskit.QuantumCircuit(qr_control, qr_free, qr_target)

        qc_mcx_qiskit.mcx(control_qubits=qr_control, target_qubit=qr_target)

        op_mcx_no_ancilla = Operator(qc_mcx_no_ancilla).data
        op_mcx_qiskit = Operator(qc_mcx_qiskit).data

        self.assertTrue(np.allclose(op_mcx_qiskit, op_mcx_no_ancilla))


    def test_mcx(self):
        """
            Comparison of 8-qubit MCX decomposition using ancilla
            with Qiskit mcx with additional unused qubit:
            - operator matrices
            - # of operators (u and cx)
        """

        n_controls = 8
        mcx_ancilla = qiskit.QuantumCircuit(n_controls + 2)

        # implementation of He et al. (2017) mcx decomposition
        mcx_ancilla.compose(mcx(n_controls), inplace=True)

        qr_mcx_qiskit_control = qiskit.QuantumRegister(n_controls, 'control')
        qr_mcx_qiskit_target = qiskit.QuantumRegister(1, 'target')
        qr_mcx_qiskit_empty = qiskit.QuantumRegister(1, 'empty')

        # Qiskit implementation of mcx
        mcx_qiskit = qiskit.QuantumCircuit(qr_mcx_qiskit_control, \
                                           qr_mcx_qiskit_target, \
                                           qr_mcx_qiskit_empty)

        mcx_qiskit.mcx(control_qubits=qr_mcx_qiskit_control, target_qubit=qr_mcx_qiskit_target)

        op_mcx_ancilla = Operator(mcx_ancilla).data
        op_mcx_qiskit = Operator(mcx_qiskit).data

        tr_mcx_ancilla = qiskit.transpile(mcx_ancilla, basis_gates=['u', 'cx'])
        tr_mcx_qiskit = qiskit.transpile(mcx_qiskit, basis_gates=['u', 'cx'])

        tr_mcx_ancilla_ops = tr_mcx_ancilla.count_ops()
        tr_mcx_qiskit_ops = tr_mcx_qiskit.count_ops()

        self.assertTrue(np.allclose(op_mcx_qiskit, op_mcx_ancilla))
        self.assertTrue(tr_mcx_ancilla_ops['u'] < tr_mcx_qiskit_ops['u'])
        self.assertTrue(tr_mcx_ancilla_ops['cx'] < tr_mcx_qiskit_ops['cx'])
