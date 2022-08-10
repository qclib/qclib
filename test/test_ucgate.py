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

""" Test creation of uniformly controlled gates """

from unittest import TestCase
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info import Operator
#from qiskit.extensions.quantum_initializer.uc import UCGate
from qclib.gates.uc_gate import UCGate

class TestUCGate(TestCase):
    """ Testing qclib.gates.ucgate """
    def test_inverse_ucg(self):
        """ "Test inverse function of uniformly controlled gates"""
        gates = [random_unitary(2, seed=42+s).data for s in range(2**2)]
        num_con = int(np.log2(len(gates)))
        reg = QuantumRegister(num_con + 1)
        circuit = QuantumCircuit(reg)

        ucg = UCGate(gates, up_to_diagonal=True)
        circuit.append(ucg, reg[1:]+ [reg[0]])
        #qc.uc(gates, q[1:], q[0], up_to_diagonal=False)
        circuit.append(circuit.inverse(), circuit.qubits)

        unitary = Operator(circuit).data
        unitary_desired = np.identity(2**circuit.num_qubits)

        self.assertTrue(np.allclose(unitary_desired, unitary))
