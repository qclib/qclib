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

'''
Quantum Image Watermarking Algorithm Based
on Haar Wavelet Transform
WEN-WEN HU, RI-GUI ZHOU, AHMED EL-RAFEI,
AND SHE-XIANG JIANG (2019)
https://ieeexplore.ieee.org/document/8812626
'''

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import HGate

class Qhwt(Gate):
    '''
    Quantum Haar Wavelet Transform (QHWT)
    '''
    def __init__(self, num_qubits, levels=None):
        self.register = QuantumRegister(num_qubits)
        self.levels = levels if levels is not None else num_qubits

        super().__init__('qhwt', num_qubits, [], "Qhwt")

    def _define(self):

        self.definition = QuantumCircuit(self.register)
        num_qubits = len(self.register)

        for level in range(self.levels):
            h_gate = HGate()
            if level > 0:
                h_gate = h_gate.control(level)

            self.definition.compose(
                h_gate,
                self.register[:level+1]
            )

            if level < num_qubits - 1:
                p_gate = self._permutation_gate(num_qubits - level)
                if level > 0:
                    p_gate = p_gate.control(level)

                self.definition.compose(
                    p_gate,
                    self.register
                )


    @staticmethod
    def _permutation_gate(num_qubits):
        gate = QuantumCircuit(num_qubits)

        for qubit in range(num_qubits-1):
            gate.swap(qubit, qubit+1)

        return gate

    @staticmethod
    def qhwt(
        circuit,
        levels,
        qubits
    ):
        '''
        Quantum Haar Wavelet Transform (QHWT)
        https://ieeexplore.ieee.org/document/8812626
        '''
        circuit.append(
            Qhwt(
                len(qubits),
                levels
            ),
            qubits
        )
