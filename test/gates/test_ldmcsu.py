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
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import Operator
from qiskit.extensions import UnitaryGate
from scipy.stats import unitary_group
from qclib.gates.ldmcsu import LdMcSpecialUnitary, Ldmcsu
from qclib.gates.util import u2_to_su2
from qclib.util import get_cnot_count

NUM_CTRL = 6


def _generate_su_2():
    """
    Returns random SU(2) matrix
    """
    u_2 = unitary_group.rvs(2)
    su_2, _ = u2_to_su2(u_2)
    return su_2


class TestLcMcSpecialUnitary(TestCase):
    """
        Test cases for the decomposition of
        Multicontrolled Special Unitary with linear depth
        by Barenco et al.
    """

    def _build_qiskit_circuit(self, su2, num_controls, ctrl_state=None):

        su2_gate = UnitaryGate(su2)
        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(num_controls + 1)
        qiskit_circ.append(su2_gate.control(num_controls, ctrl_state=ctrl_state),
                           [*controls_list, target])
        return qiskit_circ

    def _compute_bound(self, num_qubits):
        if num_qubits % 2 == 0:
            return 28 * num_qubits - 88

        return 28 * num_qubits - 92

    def test_lcmcsu_op_for_trivial_control_state(self):
        """
        Test LdMcSpecialUnitary open controls
        """
        su2 = _generate_su_2()

        num_controls = NUM_CTRL
        ldmcsu_circ = LdMcSpecialUnitary(su2, num_controls).definition
        qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_all_zero_control_states(self):
        """
        Test LdMcSpecialUnitary with open controls
        """
        su_2 = _generate_su_2()

        num_controls = NUM_CTRL
        ctrl_state = '0' * num_controls
        ldmcsu_circ = LdMcSpecialUnitary(su_2, num_controls, ctrl_state=ctrl_state).definition
        qiskit_circ = self._build_qiskit_circuit(su_2, num_controls, ctrl_state=ctrl_state)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_cnot_count(self):
        """
        Test LdMcSpecialUnitary cx count
        """
        su_2 = _generate_su_2()
        for num_controls in range(8, 10):
            ldmcsu_circ = LdMcSpecialUnitary(su_2, num_controls).definition
            ldmcsu_count = get_cnot_count(ldmcsu_circ)

            self.assertLessEqual(ldmcsu_count, self._compute_bound(num_controls + 1))


class TestMcSpecialUnitary(TestCase):
    """
        Test cases for the decomposition of
        Multicontrolled Special Unitary with linear depth
        https://arxiv.org/pdf/2302.06377.pdf
    """

    def _build_qiskit_circuit(self, su2, num_controls, ctrl_state=None):

        su2_gate = UnitaryGate(su2)
        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(num_controls + 1)
        qiskit_circ.append(su2_gate.control(num_controls, ctrl_state=ctrl_state),
                           [*controls_list, target])
        return qiskit_circ

    def _compute_bound(self, num_qubits):
        if num_qubits % 2 == 0:
            return 28 * num_qubits - 88

        return 28 * num_qubits - 92

    def test_lcmcsu_op_for_trivial_control_state(self):
        """
        Test LdMcSpecialUnitary open controls
        """
        su2 = _generate_su_2()

        num_controls = NUM_CTRL
        ldmcsu_circ = Ldmcsu(su2, num_controls).definition
        qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_all_zero_control_states(self):
        """
        Test LdMcSpecialUnitary with open controls
        """
        su_2 = _generate_su_2()

        num_controls = NUM_CTRL
        ctrl_state = '0' * num_controls
        ldmcsu_circ = Ldmcsu(su_2, num_controls, ctrl_state=ctrl_state).definition
        qiskit_circ = self._build_qiskit_circuit(su_2, num_controls, ctrl_state=ctrl_state)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_cnot_count(self):
        """
        Test LdMcSpecialUnitary cx count
        """
        su_2 = _generate_su_2()
        for num_controls in range(8, 10):
            ldmcsu_circ = Ldmcsu(su_2, num_controls).definition
            ldmcsu_count = get_cnot_count(ldmcsu_circ)

            self.assertTrue(ldmcsu_count <= 20 * (num_controls + 1) - 38)

    def test_lcmcsu_cnot_count_real_diagonal(self):
        """
        Test LdMcSpecialUnitary cx count
        """
        su_2 = RXGate(0.3).to_matrix()
        for num_controls in range(8, 10):
            ldmcsu_circ = Ldmcsu([su_2], num_controls).definition
            ldmcsu_count = get_cnot_count(ldmcsu_circ)

            self.assertTrue(ldmcsu_count <= 16 * (num_controls + 1) - 40)

    def _build_qiskit_circuit_2target(
            self,
            unitaries,
            num_controls,
            ctrl_state=None
    ):
        """"default mode: target = controls"""

        su2_gate1 = UnitaryGate(unitaries[0])
        su2_gate2 = UnitaryGate(unitaries[1])
        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(num_controls + 2)
        qiskit_circ.append(su2_gate1.control(num_controls, ctrl_state=ctrl_state),
                           [*controls_list, target])
        qiskit_circ.append(su2_gate2.control(num_controls, ctrl_state=ctrl_state),
                           [*controls_list, target + 1])
        return qiskit_circ

    def test_lcmcsu_2targets(self):
        """

        """
        num_controls = 6
        num_target_qubit = 2
        unitaries = [RXGate(0.3).to_matrix(), RXGate(0.3).to_matrix()]

        qiskit_circ = self._build_qiskit_circuit_2target(unitaries, num_controls)
        ldmcsu_circ = Ldmcsu(unitaries, num_controls, num_target=num_target_qubit).definition

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_exception_unitary(self):
        """
            Test Ldmcsu with Z gate
        """
        su2 = np.array([[-1., 0.], [0., -1.]])

        num_controls = 6
        ldmcsu_circ = Ldmcsu(su2, num_controls).definition
        qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))

    def test_lcmcsu_op_for_exception_unitary_2(self):
        """
        Test Ldmcsu diagonal gate
        """
        su2 = np.array([[np.e ** (-1j * 0.3), 0], [0, np.e ** (1j * 0.3)]])

        num_controls = 6
        ldmcsu_circ = Ldmcsu(su2, num_controls).definition
        qiskit_circ = self._build_qiskit_circuit(su2, num_controls)

        ldmcsu_op = Operator(ldmcsu_circ).data
        qiskit_op = Operator(qiskit_circ).data

        self.assertTrue(np.allclose(ldmcsu_op, qiskit_op))
