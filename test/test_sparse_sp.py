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
from qclib.state_preparation import CvoqramInitialize, PivotInitialize
from qclib.util import double_sparse, get_state


class TestCvoqram(TestCase):
    """Testing qclib.state_preparation.cvoqram"""

    def test_cvoqram(self):
        """Testing cvoqram 2 amplitudes and with auxiliary qubits"""
        data = {"001": 1 / np.sqrt(3), "110": np.sqrt(2j / 3)}
        qc_cvoqram = CvoqramInitialize(data).definition
        state = get_state(qc_cvoqram)
        self.assertTrue(np.isclose(state[0b110000], np.sqrt(2j / 3)))
        self.assertTrue(np.isclose(state[0b001000], np.sqrt(1 / 3)))

    def test_cvoqram_without_aux(self):
        """Testing cvoqram 2 amplitudes and with auxiliary qubits"""
        data = {"001": 1j * np.sqrt(1 / 3), "110": np.sqrt(2j / 3)}
        qc_cvoqram = CvoqramInitialize(data, opt_params={"with_aux": False}).definition
        state = get_state(qc_cvoqram)
        self.assertTrue(np.isclose(state[0b1100], np.sqrt(2j / 3)))
        self.assertTrue(np.isclose(state[0b0010], 1j * np.sqrt(1 / 3)))

    def test_cvoqram_random(self):
        """Testing cvoqram 4 amplitudes and with auxiliary qubits"""
        n_qubits = 4
        log_npatterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_npatterns, prob)
        data = {"".join(map(str, b)): d for b, d in data}

        qc_cvoqram = CvoqramInitialize(data).definition
        state = get_state(qc_cvoqram)
        for pattern, amp in data.items():
            index = pattern + n_qubits * "0"  # padding work qubits

            self.assertTrue(np.isclose(state[int(index, 2)], amp))

    def test_double_sparse(self):
        """Test double sparse random generation"""
        n_qubits = 6
        log_npatterns = 4
        prob = 0.8
        data = double_sparse(n_qubits, log_npatterns, prob)
        norm = 0
        for k, _ in enumerate(data):
            norm = norm + np.abs(data[k][1]) ** 2
            self.assertTrue(np.sum(data[k][0]) <= np.sum(data[k][0]))

        self.assertTrue(np.isclose(norm, 1))


class TestPivotingSP(TestCase):
    """Testing pivot state preparation"""

    def test_pivoting_sp(self):
        """Testing pivot state preparation with 2 amplitudes"""
        data = {"001": 1 / np.sqrt(3), "110": np.sqrt(2j / 3)}
        circuit = PivotInitialize(data).definition
        state = get_state(circuit)
        self.assertTrue(np.isclose(state[0b001], 1 / np.sqrt(3)))
        self.assertTrue(np.isclose(state[0b110], np.sqrt(2j / 3)))

    def test_pivotsp_random(self):
        """Testing pivot state preparation with 4 amplitudes"""
        n_qubits = 4
        log_npatterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_npatterns, prob)
        data = {"".join(map(str, b)): d for b, d in data}

        circuit = PivotInitialize(data).definition
        state = get_state(circuit)
        for pattern, amp in data.items():
            self.assertTrue(np.isclose(state[int(pattern, 2)], amp))

    def test_pivotsp_random_aux(self):
        """Testing pivot state preparation with 4 amplitudes and auxiliary qubits"""
        n_qubits = 4
        log_npatterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_npatterns, prob)
        data = {"".join(map(str, b)): d for b, d in data}

        circuit = PivotInitialize(data, opt_params={"aux": True}).definition
        state = get_state(circuit)
        for pattern, amp in data.items():
            index = pattern + "0"  # padding work qubits

            self.assertTrue(np.isclose(state[int(index, 2)], amp))
