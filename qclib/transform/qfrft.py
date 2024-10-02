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
Circuit of Quantum Fractional Fourier Transform
Zhao, Tieyu, and Yingying Chi. 2023.
https://doi.org/10.3390/fractalfract7100743
'''

from numpy import pi
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import QFT, UnitaryGate

class Qfrft(Gate):
    '''
    Quantum Fractional Fourier Transform (QFrFT)
    '''
    def __init__(self, num_targets, alpha):
        self.alpha = alpha
        self.controls = QuantumRegister(2)
        self.targets = QuantumRegister(num_targets)

        super().__init__('qfrft', num_targets+2, [], "Qfrft")

    def _define(self):
        if len(self.num_qubits) > 0:
            self.definition = QuantumCircuit(self.controls, self.targets)

            num_targets = len(self.targets)

            qft_controls = QFT(2)
            qft_targets = QFT(num_targets)

            qft_targets_2 = UnitaryGate(qft_targets @ qft_targets)

            qft_controls_inv = qft_controls.inverse()
            qft_targets_inv = qft_targets.inverse()

            qft_targets_2_inv = qft_targets_2.inverse()

            gate = QuantumCircuit(self.num_qubits, name="Qfrft")

            gate.h(self.controls[0])
            gate.h(self.controls[1])

            gate.append(
                qft_targets.control(1),
                [self.controls[1], *self.targets]
            )
            gate.append(
                qft_targets_2.control(1),
                [self.controls[0], *self.targets]
            )

            gate.append(
                qft_controls_inv,
                self.controls
            )
            gate.rz(-pi * self.alpha, self.controls[0])
            gate.rz(-pi * self.alpha / 2, self.controls[1])
            gate.append(
                qft_controls,
                self.controls
            )

            gate.append(
                qft_targets_inv.control(1),
                [self.controls[1], *self.targets]
            )
            gate.append(
                qft_targets_2_inv.control(1),
                [self.controls[0], *self.targets]
            )

            gate.h(self.controls[0])
            gate.h(self.controls[1])

            self.definition.append(
                gate,
                [*self.controls, *self.targets]
            )

        else:
            self.definition = QuantumCircuit(self.num_qubits)

    @staticmethod
    def qfrft(
        circuit,
        alpha,
        controls,
        targets
    ):
        '''
        Quantum Fractional Fourier Transform (QFrFT)
        https://doi.org/10.3390/fractalfract7100743
        '''
        circuit.append(
            Qfrft(
                alpha,
                len(targets)
            ),
            [*controls, *targets]
        )
