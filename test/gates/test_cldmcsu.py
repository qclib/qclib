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

""" Test qclib.gate.mcg.LdMcSpecialUnitary """

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXGate, RYGate, RZGate
from qiskit.quantum_info import Operator
from qclib.gates.cldmcsu import Cldmcsu



class TestCMcSpecialUnitary(TestCase):
    """
        Test cases for the decomposition of
        Multicontrolled Special Unitary with linear depth
        https://arxiv.org/pdf/2302.06377.pdf
    """

    def _build_qiskit_circuit_2target(
            self,
            num_controls
    ):
        """"default mode: target = controls"""

        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(num_controls + 3)
        qiskit_circ.mcrx(0.7, controls_list, target)
        qiskit_circ.mcrx(0.13, controls_list, target+1)
        qiskit_circ.mcrx(0.5, controls_list, target+2)

        return qiskit_circ

    def test_clcmcsu_2targets(self):
        """

        """
        num_controls = 7
        num_target_qubit = 3
        unitaries = [RXGate(0.7).to_matrix(), RXGate(0.13).to_matrix(), RXGate(0.5).to_matrix()]

        qiskit_circ = self._build_qiskit_circuit_2target(num_controls)
        cldmcsu_circ = Cldmcsu(unitaries, num_controls, num_target=num_target_qubit).definition

        qiskitt = transpile(qiskit_circ, basis_gates=['u', 'cx'], optimization_level=3)
        cldmcsut = transpile(cldmcsu_circ, basis_gates=['u', 'cx'], optimization_level=3)

        qiskit_ops = qiskitt.count_ops()
        qclib_ops = cldmcsut.count_ops()
        self.assertTrue(qiskit_ops['cx'] > qclib_ops['cx'])

        cldmcsu_op = Operator(cldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(cldmcsu_op, qiskit_op))

