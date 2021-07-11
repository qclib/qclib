"""
State preparation using Schmidt decomposition 	arXiv:1003.5760
"""
import numpy as np
import qiskit
from qclib.unitary import unitary
from qclib.util import get_state


def initialize(unit_vector):
    """
    State preparation using Schmidt decomposition arXiv:1003.5760
    """
    state = np.copy(unit_vector)

    size = len(state)
    n_qubits = np.log2(size)
    r = n_qubits % 2

    state.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + r)))

    u, d, v = np.linalg.svd(state)
    d = d / np.linalg.norm(d)

    A = qiskit.QuantumRegister(n_qubits//2 + r)
    B = qiskit.QuantumRegister(n_qubits//2)

    sp_circuit = qiskit.QuantumCircuit(A, B)

    if len(d) > 2:
        circ = initialize(d)
        sp_circuit.append(circ, B)
    else:
        sp_circuit.initialize(d, B)

    for k in range(int(n_qubits//2)):
        sp_circuit.cx(B[k], A[k])

    # apply gate U to the first register
    gate_u = unitary(u, 'qsd')
    sp_circuit.compose(gate_u, B, inplace=True)

    # apply gate V to the second register
    gate_v = unitary(v.T, 'qsd')
    sp_circuit.compose(gate_v, A, inplace=True)

    return sp_circuit


def pivoting(index_zero, index_nonzero, target_size, state=None):
    """

    Attributes
    ----------
    index_zero:
    index_nonzero:
    target_size:
    state:

    Returns:

    """
    n_qubits = len(index_zero)
    target = qiskit.QuantumRegister(n_qubits - target_size, name='t')
    remainder = qiskit.QuantumRegister(target_size, name='r')
    # circuit = qiskit.QuantumCircuit(target, remainder)
    circuit = qiskit.QuantumCircuit(remainder, target)

    target_nonzero = index_nonzero[:n_qubits-target_size][::-1]
    target_zero = index_zero[:n_qubits-target_size][::-1]
    remainder_nonzero = index_nonzero[n_qubits-target_size:][::-1]
    remainder_zero = index_zero[n_qubits-target_size:][::-1]

    index_differ = 0
    for index, value in enumerate(target_nonzero):
        if target_nonzero[index] != target_zero[index]:
            index_differ = index
            break

    for index, value in enumerate(target_nonzero):
        if index_differ != index:
            if target_nonzero[index] != target_zero[index]:
                circuit.cx(target[index_differ], target[index])



    for index, value  in enumerate(remainder_nonzero):
        if remainder_nonzero[index] != remainder_zero[index]:
            circuit.cx(target[index_differ], remainder[index])

    for index, value in enumerate(remainder_zero):
        if value == '0':
            circuit.x(remainder[index])

    circuit.mcx(remainder, target[index_differ])

    for index, value in enumerate(remainder_zero):
        if value == '0':
            circuit.x(remainder[index])

    next_state = None
    if list(state):
        # next_state should not be calculated using a quantum circuit
        circ_next = initialize(state / np.linalg.norm(state))
        circ_next.compose(circuit, circ_next.qubits, inplace=True)
        next_state = get_state(circ_next)
    return circuit, next_state


def sparse_initialize(state):

    n_qubits = np.log2(len(state))
    n_qubits = int(n_qubits)

    pivot_circuit = qiskit.QuantumCircuit(n_qubits)

    non_zero = _count_nonzero(state)

    target_size = np.log2(non_zero)
    target_size = np.ceil(target_size)
    target_size = int(target_size)

    fim = _count_nonzero(state[:2**target_size])
    # for k, value in enumerate(state):
    #     if not np.isclose(value, 0.0) and k < target_size:
    #         fim += 1

    next_state = state.copy()
    while fim != non_zero:
        for index, value in enumerate(next_state):
            if not np.isclose(value, 0):
                continue

            index_zero = index
            break

        for index2, value2 in enumerate(next_state[2**target_size:]):
            if not np.isclose(value2, 0):
                fim += 1
                index_nonzero = index2 + 2**target_size

                txt = "{0:0" + str(n_qubits) + "b}"
                index_zero_bin = txt.format(index_zero)
                index_nonzero_bin = txt.format(index_nonzero)
                print('pivoting')
                circ, next_state = pivoting(index_zero_bin, index_nonzero_bin, target_size, next_state)
                next_state[next_state < 1e-3] = 0.0
                pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)
                break

    initialize_circ = initialize(next_state[:2**target_size])
    sp_circuit = qiskit.QuantumCircuit(n_qubits)
    sp_circuit.compose(initialize_circ, sp_circuit.qubits[:target_size], inplace=True)
    sp_circuit.compose(pivot_circuit.reverse_ops(), sp_circuit.qubits, inplace=True)

    return sp_circuit


def _count_nonzero(state):
    non_zero = 0
    for k in state:
        if not np.isclose(k, 0.0):
            non_zero += 1
    return non_zero
