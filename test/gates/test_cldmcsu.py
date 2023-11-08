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
from qiskit import QuantumRegister, QuantumCircuit, transpile
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import Operator
from qclib.gates.multitargetmcsu2 import MultiTargetMCSU2



class TestCMcSpecialUnitary(TestCase):
    """
        Test cases for the decomposition of
        Multicontrolled Special Unitary with linear depth
        https://arxiv.org/pdf/2302.06377.pdf
    """

    def _build_qiskit_circ(
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

    def _build_ldmcsu_circ(self, unitary_list, num_controls):
        """"
        default mode: target = controls
        """

        controls_list = list(range(num_controls))
        target = num_controls
        ldmcsu_circ = QuantumCircuit(num_controls + 3)
        MultiTargetMCSU2.multi_target_mcsu2(ldmcsu_circ, unitary_list[0], controls_list, target)
        MultiTargetMCSU2.multi_target_mcsu2(ldmcsu_circ, unitary_list[1], controls_list, target + 1)
        MultiTargetMCSU2.multi_target_mcsu2(ldmcsu_circ, unitary_list[2], controls_list, target + 2)

        return ldmcsu_circ

    def _build_cldmcsu_circ(self, unitary_list, num_controls):
        """"
        default mode: target = controls
        """

        controls = QuantumRegister(num_controls)
        target = QuantumRegister(len(unitary_list))
        cldmcsu_circ = QuantumCircuit(controls, target)
        MultiTargetMCSU2.multi_target_mcsu2(cldmcsu_circ, unitary_list, controls, target)

        return cldmcsu_circ

    def test_clcmcsu_3targets(self):
        """
        Test for comparison of a cascade uf 3 multi-controlled SU(2) using
        qiskit and cldmcsu implementations.
        """
        num_controls = 7
        unitary_list = [RXGate(0.7).to_matrix(), RXGate(0.13).to_matrix(), RXGate(0.5).to_matrix()]

        qiskit_circ = self._build_qiskit_circ(num_controls)
        cldmcsu_circ = self._build_cldmcsu_circ(unitary_list, num_controls)

        qiskitt = transpile(qiskit_circ, basis_gates=['u', 'cx'], optimization_level=3)
        cldmcsut = transpile(cldmcsu_circ, basis_gates=['u', 'cx'], optimization_level=3)

        qiskit_ops = qiskitt.count_ops()
        qclib_ops = cldmcsut.count_ops()
        self.assertTrue(qiskit_ops['cx'] > qclib_ops['cx'])

        cldmcsu_op = Operator(cldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(cldmcsu_op, qiskit_op))
