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
This module provides access to functions to
implement generic quantum computations.
"""

from math import ceil, log2
import scipy as sp
import qiskit
from qiskit import transpile
from qiskit.circuit.library import CXGate
from qiskit.extensions import UnitaryGate
from qiskit.extensions import UCRYGate, UCRZGate
from qiskit.extensions.quantum_initializer import UCGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix

import numpy as np


def unitary(gate, decomposition='qsd', iso=False):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(gate)
    if size > 4:
        n_qubits = int(log2(size))

        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)

        right_gates, theta, left_gates = \
            sp.linalg.cossin(gate, size/2, size/2, separate=True)

        if iso:
            gate_left = unitary(left_gates[0])
            circuit = circuit.compose(gate_left, qubits[:-1])
        else:
            gate_left = _unitary(list(left_gates), n_qubits, decomposition)
            circuit = circuit.compose(gate_left, qubits)

        gate_right = _unitary(list(right_gates), n_qubits, decomposition)

        circuit.append(UCRYGate(list(2 * theta)), [n_qubits-1] + list(range(n_qubits-1)))

        circuit = circuit.compose(gate_right, qubits)

        return circuit

    circuit = qiskit.QuantumCircuit(int(log2(size)))
    circuit.append(UnitaryGate(gate), circuit.qubits)

    return circuit


def _unitary(gate_list, n_qubits, decomposition='qsd'):

    if len(gate_list[0]) == 2:
        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)
        circuit.append(UCGate(gate_list), qubits[[0]] + qubits[1:])
        return circuit

    if decomposition == 'csd':
        return _csd(gate_list, n_qubits)

    # QSD
    return _qsd(*gate_list)


# CSD decomposition


def _csd(gate_list, n_qubits):
    left, mid, right = _multiplexed_csd(gate_list)

    gate_left = _unitary(left, n_qubits)
    gate_right = _unitary(right, n_qubits)

    qubits = qiskit.QuantumRegister(n_qubits)
    circuit = qiskit.QuantumCircuit(qubits)

    circuit = circuit.compose(gate_left, qubits)

    target = int(n_qubits - log2(len(left)))
    control = list(range(0, target)) + list(range(target + 1, n_qubits))
    circuit.append(UCRYGate(list(mid)), [target] + control)

    circuit = circuit.compose(gate_right, qubits)

    return circuit


def _multiplexed_csd(gate_list):
    left = []
    mid = []
    right = []
    size = len(gate_list[0])
    for gate in gate_list:
        right_gates, theta, left_gates = \
            sp.linalg.cossin(gate, size / 2, size / 2, separate=True)

        left = left + list(left_gates)
        right = right + list(right_gates)
        mid = mid + list(2 * theta)

    return left, mid, right


def _qsd(gate1, gate2):
    n_qubits = int(log2(len(gate1))) + 1

    list_d, gate_v, gate_w = _compute_gates(gate1, gate2)

    left_gate = unitary(gate_w, 'qsd')
    right_gate = unitary(gate_v, 'qsd')

    qubits = qiskit.QuantumRegister(n_qubits)
    circuit = qiskit.QuantumCircuit(qubits)

    circuit = circuit.compose(left_gate, qubits[0:-1])
    circuit.append(UCRZGate(list(-2*np.angle(list_d))), qubits[[-1]] + qubits[0:-1])
    circuit = circuit.compose(right_gate, qubits[0:-1])
    return circuit


def _closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


def _compute_gates(gate1, gate2):

    d_square, gate_v = np.linalg.eig(gate1 @ gate2.conj().T)
    list_d = np.sqrt(d_square, dtype=complex)
    gate_d = np.diag(list_d)

    if not is_unitary_matrix(gate_v):
        # degeneracy
        gate_v = _closest_unitary(gate_v)

    gate_w = gate_d @ gate_v.conj().T @ gate2

    return list_d, gate_v, gate_w


def cnot_count(gate, decomposition='csd', method='estimate'):
    """
    Count the number of CNOTs to decompose the unitary.
    """
    if method == 'estimate':
        return _cnot_count_estimate(gate, decomposition)

    # Exact count
    circuit = unitary(gate, decomposition)
    transpiled_circuit = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'],
                                   optimization_level=0)
    count_ops = transpiled_circuit.count_ops()
    if 'cx' in count_ops:
        return count_ops[CXGate()]

    return 0


def _cnot_count_estimate(gate, decomposition='csd'):
    """
    Estimate the number of CNOTs to decompose the unitary.
    """
    n_qubits = int(log2(gate.shape[0]))
    if n_qubits == 1:
        return 0

    if decomposition == 'csd':
        # Table 1 from "Synthesis of Quantum Logic Circuits", Shende et al.
        return int(ceil(4**n_qubits - 2*2**n_qubits))

    # Upper-bound expression for the unitary decomposition QSD l=2 without the optimizations.
    # With the optimizations, they need to be replaced.
    # Table 1 from "Synthesis of Quantum Logic Circuits", Shende et al.
    return int(ceil(9/16*2**(2*n_qubits) - 3/2 * 2**n_qubits))
