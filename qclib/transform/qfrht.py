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

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import SXGate
from qclib.transform import Qfrft


'''
Non-fractional quantum Hartley transform:
On the Irresistible Efficiency of Signal Processing
Methods in Quantum Computing
Andreas Klappenecker, and Martin Rötteler, 2001.
https://arxiv.org/abs/quant-ph/0111039
'''

class Qfrht(Gate):
    '''
    Quantum Fractional Hartley Transform (QFrFT)
    '''
    def __init__(self, num_targets, alpha):
        self.alpha = alpha
        self.controls = QuantumRegister(3)
        self.targets = QuantumRegister(num_targets)

        super().__init__('qfrht', num_targets+3, [], "QFrHT")

    def _define(self):

        self.definition = QuantumCircuit(self.controls, self.targets)

        if len(self.targets) > 0:
            num_targets = len(self.targets)

            qfrft = Qfrft(num_targets, alpha=self.alpha).to_gate()

            gate = QuantumCircuit(self.controls, self.targets, name="QFrHT")

            gate.h(self.controls[0])

            gate.append(
                qfrft,
                [*self.controls[1:], *self.targets]
            )
            gate.append(
                qfrft.control(1),
                [*self.controls, *self.targets]
            )
            gate.append(
                qfrft.control(1),
                [*self.controls, *self.targets]
            )

            gate.append(SXGate().inverse(), [self.controls[0]])

            gate.append(
                qfrft.control(1),
                [*self.controls, *self.targets]
            )
            gate.append(
                qfrft.control(1),
                [*self.controls, *self.targets]
            )

            gate.h(self.controls[0])

            self.definition.append(
                gate,
                [*self.controls, *self.targets]
            )

    @staticmethod
    def qfrht(
        circuit,
        alpha,
        controls,
        targets
    ):
        '''
        Quantum Fractional Hartley Transform (QFrHT)

        Non-fractional quantum Hartley transform:
        https://arxiv.org/abs/quant-ph/0111039
        '''
        circuit.append(
            Qfrht(
                len(targets),
                alpha
            ),
            [*controls, *targets]
        )
