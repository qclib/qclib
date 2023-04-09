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
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.stats import unitary_group
from qclib.gates.ldmcsu import LdMcSpecialUnitary
from qclib.gates.util import u2_to_su2
from qclib.util import get_cnot_count

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=maybe-no-member

class TestLcMcSpecialUnitary(TestCase):
    """
        Test cases for the decomposition of
        Multicontrolled Special Unitary with linear depth
        by Barenco et al.
    """

    def _generate_su_2(self):
        u_2 = unitary_group.rvs(2)
        su_2, _ = u2_to_su2(u_2)
        return su_2

    def _build_qiskit_circuit(self, su2, num_controls, ctrl_state=None):

        su2_gate = QuantumCircuit(1)
        su2_gate.unitary(su2, 0)
        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(num_controls + 1)
        qiskit_circ.append(su2_gate.control(num_controls, ctrl_state=ctrl_state),
                           [*controls_list, target])
        return qiskit_circ

    def _compute_bound(self, num_qubits):
        if num_qubits % 2 == 0:
            return 28 * num_qubits - 88
        else:
            return 28 * num_qubits - 92


    def test_lcmcsu_op_for_trivial_control_state(self):
        su2 = self._generate_su_2()

        for num_controls in range(1, 9):
            ldmcsu_circ = LdMcSpecialUnitary(su2, num_controls).definition
            qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

            ldmcsu_op = Operator(ldmcsu_circ).data
            qiskit_op = Operator(qiskit_circ).data

            self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_all_zero_control_states(self):
        su_2 = self._generate_su_2()

        for num_controls in range(1, 9):
            ctrl_state = '0' * num_controls
            ldmcsu_circ = LdMcSpecialUnitary(su_2, num_controls, ctrl_state=ctrl_state).definition
            qiskit_circ = self._build_qiskit_circuit(su_2, num_controls, ctrl_state=ctrl_state)

            ldmcsu_op = Operator(ldmcsu_circ).data
            qiskit_op = Operator(qiskit_circ).data

            self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_cnot_count(self):
        su_2 = self._generate_su_2()
        for num_controls in range(8, 10):
            ldmcsu_circ = LdMcSpecialUnitary(su_2, num_controls).definition
            ldmcsu_count = get_cnot_count(ldmcsu_circ)

            self.assertLessEqual(ldmcsu_count, self._compute_bound(num_controls+1))

    def test_lcmcsu_op_for_exception_unitary(self):
        su2 = np.array([[-1, 0], [0, -1]])

        for num_controls in range(1, 9):
            ldmcsu_circ = LdMcSpecialUnitary(su2, num_controls).definition
            qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

            ldmcsu_op = Operator(ldmcsu_circ).data
            qiskit_op = Operator(qiskit_circ).data

            self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_exception_unitary_2(self):
        su2 = np.array([[np.e**(-1j*0.3), 0], [0, np.e**(1j*0.3)]])

        for num_controls in range(1, 9):
            ldmcsu_circ = LdMcSpecialUnitary(su2, num_controls).definition
            qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

            ldmcsu_op = Operator(ldmcsu_circ).data
            qiskit_op = Operator(qiskit_circ).data

            self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))