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

"""
Black-box state preparation
Grover, Lov K. "Synthesis of quantum superpositions by quantum computation."
Physical review letters 85.6 (2000): 1334.

Gate U2 in PRL 85.6 (2000) is implemented with uniformly controlled rotations
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.extensions import UCRYGate, UCRZGate
from qiskit.extensions import UnitaryGate
from qclib.gates.initialize import Initialize


class BlackBoxInitialize(Initialize):
    """
    Black-box state preparation
    Grover, Lov K. "Synthesis of quantum superpositions by quantum computation."
    Physical review letters 85.6 (2000): 1334.

    Gate U2 in PRL 85.6 (2000) is implemented with uniformly controlled rotations
    """

    def __init__(self, params, label=None):
        self._name = "blackbox"
        self._get_num_qubits(params)
        self.num_qubits += 1

        if label is None:
            label = "BBSP"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        n_amplitudes = len(self.params)

        theta = 2 * np.arccos(np.abs(self.params))
        phi = -2 * np.angle(self.params)

        ury_gate = UCRYGate(list(theta))
        urz_gate = UCRZGate(list(phi))

        gate_u = QuantumCircuit(self.num_qubits, name="U")
        gate_u.h(gate_u.qubits[1:])
        gate_u.append(ury_gate, gate_u.qubits)
        gate_u.append(urz_gate, gate_u.qubits)
        gate_u = gate_u.to_instruction()

        it_matrix = [[-1, 0], [0, 1]]
        gate_it = UnitaryGate(it_matrix)
        gate_it.name = "I_t"

        gate_is = gate_it.control(self.num_qubits - 1, ctrl_state=0)
        gate_is.name = "I_s"

        repetitions = (np.pi / 4) * (
            np.sqrt(n_amplitudes) / np.linalg.norm(self.params)
        )
        repetitions = int(repetitions)

        q_circuit = QuantumCircuit(self.num_qubits)
        for _ in range(repetitions):
            q_circuit.append(gate_u, q_circuit.qubits)
            q_circuit.append(gate_it, q_circuit.qubits[0:1])
            q_circuit.append(gate_u.inverse(), q_circuit.qubits)
            q_circuit.append(gate_is, q_circuit.qubits)

        q_circuit.append(gate_u, q_circuit.qubits)

        if repetitions % 2 == 1:
            q_circuit.global_phase = np.pi

        return q_circuit

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
        if qubits is None:
            q_circuit.append(BlackBoxInitialize(state), q_circuit.qubits)
        else:
            q_circuit.append(BlackBoxInitialize(state), qubits)
