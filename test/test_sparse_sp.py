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

""" Test sparse initialization """

from unittest import TestCase
import numpy as np
from qiskit.quantum_info import Operator
from qclib.state_preparation import CvoqramInitialize, PivotInitialize
from qclib.util import double_sparse, get_state


class TestCvoqram(TestCase):
    """ Testing qclib.state_preparation.cvo_qram """

    def test_cvo_qram(self):
        """ Testing cvo_qram 2 amplitudes and with auxiliary qubits """
        data = {'001': 1 / np.sqrt(3), '110': np.sqrt(2j / 3)}
        qc_cvo_qram = CvoqramInitialize(data).definition
        state = get_state(qc_cvo_qram)
        self.assertTrue(np.isclose(state[0b110000], np.sqrt(2j / 3)))
        self.assertTrue(np.isclose(state[0b001000], np.sqrt(1 / 3)))

    def test_cvo_qram_without_aux(self):
        """ Testing cvo_qram 2 amplitudes and with auxiliary qubits """
        data = {'001': 1j * np.sqrt(1 / 3), '110': np.sqrt(2j / 3)}
        qc_cvo_qram = CvoqramInitialize(data, opt_params={'with_aux': False}).definition
        state = get_state(qc_cvo_qram)
        self.assertTrue(np.isclose(state[0b1100], np.sqrt(2j / 3)))
        self.assertTrue(np.isclose(state[0b0010], 1j * np.sqrt(1 / 3)))

    def test_cvo_qram_random(self):
        """ Testing cvo_qram 4 amplitudes and with auxiliary qubits """
        n_qubits = 4
        log_n_patterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_n_patterns, prob)

        qc_cvo_qram = CvoqramInitialize(data).definition
        state = get_state(qc_cvo_qram)
        for pattern, amp in data.items():
            index = pattern + n_qubits * '0'  # padding work qubits

            self.assertTrue(np.isclose(state[int(index, 2)], amp))

    def test_cvo_qram_random_without_aux(self):
        """ Testing cvo_qram 4 amplitudes and without auxiliary qubits """
        n_qubits = 4
        log_n_patterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_n_patterns, prob)

        qc_cvo_qram = CvoqramInitialize(data, opt_params={'with_aux': False}).definition
        state = get_state(qc_cvo_qram)

        for pattern, amp in data.items():
            index = pattern + '0'  # padding aux qubit

            self.assertTrue(np.isclose(state[int(index, 2)], amp))

    def test_double_sparse(self):
        """ Test double sparse random generation """
        n_qubits = 6
        log_n_patterns = 4
        prob = 0.8
        data = double_sparse(n_qubits, log_n_patterns, prob)
        norm = 0
        for _, val in data.items():
            norm = norm + np.abs(val) ** 2

        self.assertTrue(np.isclose(norm, 1))


class TestPivotingSP(TestCase):
    """ Testing pivot state preparation """

    def test_pivoting_sp(self):
        """ Testing pivot state preparation with 2 amplitudes """
        data = {'001': 1 / np.sqrt(3), '110': np.sqrt(2j / 3)}
        circuit = PivotInitialize(data).definition
        state = get_state(circuit)
        self.assertTrue(np.isclose(state[0b001], 1 / np.sqrt(3)))
        self.assertTrue(np.isclose(state[0b110], np.sqrt(2j / 3)))

    def test_inverse(self):
        data = {'001': 1 / np.sqrt(3), '110': np.sqrt(2j / 3)}
        circuit = PivotInitialize(data).definition
        circ_op = Operator(circuit).data
        inv_circ = circuit.inverse()
        inv_op = Operator(inv_circ).data
        self.assertTrue(np.allclose(circ_op @ inv_op, np.eye(8)))

    def test_pivot_sp_random(self):
        """ Testing pivot state preparation with 4 amplitudes """
        n_qubits = 4
        log_n_patterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_n_patterns, prob)

        circuit = PivotInitialize(data).definition
        state = get_state(circuit)
        for pattern, amp in data.items():
            self.assertTrue(np.isclose(state[int(pattern, 2)], amp))

    def test_pivot_sp_random_aux(self):
        """ Testing pivot state preparation with 4 amplitudes and auxiliary qubits"""
        n_qubits = 4
        log_n_patterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_n_patterns, prob)

        circuit = PivotInitialize(data, opt_params={'aux': True}).definition
        state = get_state(circuit)
        for pattern, amp in data.items():
            index = pattern + '0'  # padding work qubits

            self.assertTrue(np.isclose(state[int(index, 2)], amp))
