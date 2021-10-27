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

""" Test Probabilistic Quantum Memory"""

from unittest import TestCase
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qclib.memory.pqm import initialize as pqm
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_counts


class TestPQM(TestCase):
    """ Testing qclib.memory.pqm"""
    @staticmethod
    def _run_pqm(is_classical_pattern):
        """ run PQM with classical or quantum input"""
        # pqm memory data
        data = [0, 1, 0, 0]
        data = data / np.linalg.norm(data)

        # initialize quantum registers
        memory = QuantumRegister(2, 'm')
        aux = QuantumRegister(1, 'c')
        output = ClassicalRegister(1)
        circ = QuantumCircuit(memory, aux, output)

        # initialize data
        init_gate = initialize(data)
        circ.append(init_gate, memory)

        # initialize input pattern
        bin_input = [1, 0]

        if is_classical_pattern:
            # run pqm recovery algorithm
            pqm(circ, bin_input, memory, aux, is_classical_pattern=True)
        else:
            q_bin_input = QuantumRegister(2)
            circ.add_register(q_bin_input)

            # Pattern basis encoding
            for k, bit in enumerate(bin_input):
                if bit == 1:
                    circ.x(q_bin_input[k])
            # run pqm recovery algorithm
            pqm(circ, q_bin_input, memory, aux, is_classical_pattern=False)

        # measure output and verify results
        circ.measure(aux, output) # pylint: disable=maybe-no-member
        counts = get_counts(circ)

        return counts

    def test_classical_input(self):
        """ Testing PQM with classical input """
        # classical input pattern
        counts = TestPQM._run_pqm(True)

        self.assertTrue(counts['0'] / 1024 == 1)

    def test_quantum_input(self):
        """ Testing PQM with quantum input """
        # quantum input pattern
        counts = TestPQM._run_pqm(False)

        self.assertTrue(counts['0'] / 1024 == 1)
