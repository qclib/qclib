'''
This module provides access to functions to
implement generic quantum computations.
'''

# pylint: disable=maybe-no-member

import scipy as sp
import qiskit
import numpy as np


def _unitary(gate_list, n_qubits):

    if len(gate_list[0]) == 2:
        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)
        circuit.uc(gate_list, qubits[1:], qubits[0])
        return circuit

    left, mid, right = _multiplexed_csd(gate_list)

    gate_left = _unitary(left, n_qubits)
    gate_right = _unitary(right, n_qubits)

    qubits = qiskit.QuantumRegister(n_qubits)
    circuit = qiskit.QuantumCircuit(qubits)

    circuit = circuit.compose(gate_left, qubits)

    target = int(n_qubits - np.log2(len(left)))
    control = list(range(0, target)) + list(range(target + 1, n_qubits))
    circuit.ucry(list(mid), control, target)

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


def unitary(gate):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(gate)
    if size > 2:
        n_qubits = int(np.log2(size))

        qubits = qiskit.QuantumRegister(n_qubits)
        circuit = qiskit.QuantumCircuit(qubits)

        right_gates, theta, left_gates = \
            sp.linalg.cossin(gate, size/2, size/2, separate=True)

        gate_left = _unitary(list(left_gates), n_qubits)
        gate_right = _unitary(list(right_gates), n_qubits)

        circuit = circuit.compose(gate_left, qubits)
        circuit.ucry(list(2*theta), list(range(n_qubits-1)), n_qubits-1)
        circuit = circuit.compose(gate_right, qubits)

        return circuit

    circuit = qiskit.QuantumCircuit(1)
    circuit.unitary(gate, circuit.qubits)

    return circuit
