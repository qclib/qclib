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

""" Test majority gate """

from unittest import TestCase
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qclib.gates.majority import operate as majority
from qclib.util import get_counts


class TestMajority(TestCase):
    """Testing qclib.gate.majority"""

    @staticmethod
    def _run_majority(bin_input):
        """Run majority gate and return counts"""
        # initialize quantum registers
        controls = QuantumRegister(len(bin_input), "controls")
        target = QuantumRegister(1, "target")
        output = ClassicalRegister(1)
        circuit = QuantumCircuit(controls, target, output)

        # Pattern basis encoding
        for k, bit in enumerate(bin_input):
            if bit == 1:
                circuit.x(controls[k])

        majority(circuit, controls, target)

        # measure output and verify results
        # pylint: disable=maybe-no-member
        circuit.measure(target, output)
        counts = get_counts(circuit)

        return counts

    def _test_majority(self, n_qubits):
        """Testing majority"""
        for i in range(2**n_qubits):
            bits = [int(j) for j in "{:0{}b}".format(i, n_qubits)]
            counts = TestMajority._run_majority(bits)

            if sum(bits) >= n_qubits / 2:
                self.assertTrue(counts["1"] / 1024 == 1)
            else:
                self.assertTrue(counts["0"] / 1024 == 1)

    # Keep tests separated by number of qubits to make it easier to identify errors.
    def test_majority_2(self):
        """Testing 2 qubits majority"""
        TestMajority._test_majority(self, 2)

    def test_majority_3(self):
        """Testing 3 qubits majority"""
        TestMajority._test_majority(self, 3)

    def test_majority_4(self):
        """Testing 4 qubits majority"""
        TestMajority._test_majority(self, 4)

    def test_majority_5(self):
        """Testing 5 qubits majority"""

        TestMajority._test_majority(self, 5)

    def test_majority_6(self):
        """Testing 6 qubits majority"""
        TestMajority._test_majority(self, 6)

    def test_majority_7(self):
        """Testing 7 qubits majority"""
        TestMajority._test_majority(self, 7)
