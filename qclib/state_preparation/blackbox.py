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
Black-box state preparation (real amplitudes)
Grover, Lov K. "Synthesis of quantum superpositions by quantum computation."
Physical review letters 85.6 (2000): 1334.

Gate U2 in PRL 85.6 (2000) is implemented with uniformly controlled rotations
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.extensions import UCRYGate, UCRZGate
from qiskit.extensions import UnitaryGate

def initialize(state_vector):
    """
    Blackbox state preparation
    https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.85.1334

    For instance, to initialize the state a|00> + b|10> (|a|^2+|b|^2=1)
        $ state = [a, 0, b, 0]
        $ circuit = initialize(state)

    Parameters
    ----------
    state_vector: 2^n dimensional real unit vector

    Returns
    -------
    q_circuit: Circuit that initializes a approximated state_vector in a quantum device
    """
    n_amplitudes = len(state_vector)
    n_qubits = float(np.log2(n_amplitudes))
    if not n_qubits.is_integer():
        raise Exception("state_vector size is not a power of 2")

    theta = 2 * np.arccos(np.abs(state_vector))
    phi = -2 * np.angle(state_vector)

    ury_gate = UCRYGate(list(theta))
    urz_gate = UCRZGate(list(phi))

    n_qubits = int(n_qubits) + 1
    gate_u = QuantumCircuit(n_qubits, name='U')
    gate_u.h(gate_u.qubits[1:])
    gate_u.append(ury_gate, gate_u.qubits)
    gate_u.append(urz_gate, gate_u.qubits)
    gate_u = gate_u.to_instruction()

    it_matrix = [[-1, 0], [0, 1]]
    gate_it = UnitaryGate(it_matrix)
    gate_it.name = 'I_t'

    gate_is = gate_it.control(n_qubits-1, ctrl_state=0)
    gate_is.name = 'I_s'

    repetitions = (np.pi/4) * (np.sqrt(n_amplitudes) / np.linalg.norm(state_vector))
    repetitions = int(repetitions)

    q_circuit = QuantumCircuit(n_qubits)
    for _ in range(repetitions):
        q_circuit.append(gate_u, q_circuit.qubits)
        q_circuit.append(gate_it, q_circuit.qubits[0:1])
        q_circuit.append(gate_u.inverse(), q_circuit.qubits)
        q_circuit.append(gate_is, q_circuit.qubits)

    q_circuit.append(gate_u, q_circuit.qubits)

    if repetitions % 2 == 1:
        q_circuit.global_phase = np.pi

    return q_circuit
