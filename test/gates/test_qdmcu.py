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

""" Test qclib.gate.qdmcu.QDMCU """

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.stats import unitary_group
from qclib.gates.qdmcu import Qdmcu
from qclib.util import get_cnot_count

class TestQDMCU(TestCase): 

    def _generate_u2(self):
        return unitary_group.rvs(2)

    def _build_qiskit_qdmcu(self, u2, num_controls, ctrl_state: str = None):
        u2_gate = QuantumCircuit(1)
        u2_gate.unitary(u2, 0)
        controls_list = list(range(num_controls))
        target = num_controls
        qiskit_circ = QuantumCircuit(target + 1)
        qiskit_circ.append(u2_gate.control(num_controls, ctrl_state=ctrl_state), 
                            [*controls_list, target])
        return qiskit_circ
    
    def test_gate_trivial_ctrl_state(self):
        """Compares QDMCU result gate against qiskit's with a trivial state"""
        u2 = self._generate_u2()
        for n_ctrl in range(1, 8):
            qiskit_circ = self._build_qiskit_qdmcu(u2, n_ctrl)
            qdmcu_circ = Qdmcu(u2, n_ctrl).definition

            qiskit_op = Operator(qiskit_circ).data
            qdmcu_op  = Operator(qdmcu_circ).data

            self.assertTrue(np.allclose(qiskit_op, qdmcu_op))

    def test_gate_random_ctrl_state(self):
        """Compares QDMCU result gate against qiskit's with a trivial state"""

        u2 = self._generate_u2()
        for n_ctrl in range(1, 8):
            ctrl_state = f'{np.random.randint(1, 2**n_ctrl):0{n_ctrl}b}'
            qiskit_circ = self._build_qiskit_qdmcu(u2, n_ctrl, ctrl_state=ctrl_state)
            qdmcu_circ = Qdmcu(u2, n_ctrl, ctrl_state=ctrl_state).definition

            qiskit_op = Operator(qiskit_circ).data
            qdmcu_op  = Operator(qdmcu_circ).data

            self.assertTrue(np.allclose(qiskit_op, qdmcu_op))

    def test_cnot_count(self):
        """Compares QDMCU cnot count against qiskit's cnot count"""
        u2 = self._generate_u2()
        for n_ctrl in range(1, 8):
            ctrl_state = f'{np.random.randint(1, 2**n_ctrl):0{n_ctrl}b}'
            qiskit_circ = self._build_qiskit_qdmcu(u2, n_ctrl, ctrl_state=ctrl_state)
            qdmcu_circ = Qdmcu(u2, n_ctrl, ctrl_state=ctrl_state).definition

            qiskit_cnots = get_cnot_count(qiskit_circ)
            qdmcu_cnots  = get_cnot_count(qdmcu_circ)

            self.assertLessEqual(qdmcu_cnots, qiskit_cnots)
