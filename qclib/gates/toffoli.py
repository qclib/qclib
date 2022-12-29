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
Toffoli decomposition explained in Lemma 8 from
Quantum Circuits for Isometries.
https://arxiv.org/abs/1501.06911
'''

from numpy import pi
from qiskit import QuantumCircuit
from qiskit.circuit import Gate

class Toffoli(Gate):
    def __init__(self, cancel=None):
        self.cancel = cancel
        
        super().__init__('toffoli', 3, [], "Toffoli")

    def _define(self):
        self.definition = QuantumCircuit(3)

        theta = pi / 4.

        control_qubits = self.definition.qubits[:2]
        target_qubit = self.definition.qubits[-1]

        if self.cancel != 'left':
            self.definition.u(theta=-theta, phi=0., lam=0., qubit=target_qubit)
            self.definition.cx(control_qubits[0], target_qubit)
            self.definition.u(theta=-theta, phi=0., lam=0., qubit=target_qubit)

        self.definition.cx(control_qubits[1], target_qubit)

        if self.cancel != 'right':
            self.definition.u(theta=theta, phi=0., lam=0., qubit=target_qubit)
            self.definition.cx(control_qubits[0], target_qubit)
            self.definition.u(theta=theta, phi=0., lam=0., qubit=target_qubit)
    
    @staticmethod
    def ccx(circuit, controls=None, target=None, cancel=None):
        if controls is None or target is None:
            circuit.append(Toffoli(cancel), circuit.qubits[:3])
        else:
            circuit.append(Toffoli(cancel), [*controls, target])