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
    # index_nonzero = index_nonzero[::-1]
    # index_zero = index_zero[::-1]
    n_qubits = len(index_zero)
    target = qiskit.QuantumRegister(n_qubits - target_size, name='t')
    remainder = qiskit.QuantumRegister(target_size, name='r')
    # circuit = qiskit.QuantumCircuit(target, remainder)
    circuit = qiskit.QuantumCircuit(n_qubits)
    # circuit = qiskit.QuantumCircuit(remainder, target)

    target_nonzero = index_nonzero[:n_qubits-target_size]
    target_zero = index_zero[:n_qubits-target_size]
    remainder_nonzero = index_nonzero[n_qubits-target_size:]
    remainder_zero = index_zero[n_qubits-target_size:]

    tg = list(range(n_qubits - target_size))
    rg = list(range(n_qubits - target_size, n_qubits))

    index_differ = 0
    for k in tg:
        if index_nonzero[k] != index_zero[k]:
            index_differ = k
            ctrl_state = index_nonzero[k]
            break

    # index_differ = 0
    # for index, value in enumerate(target_nonzero):
    #     if target_nonzero[index] != target_zero[index]:
    #         index_differ = index
    #         ctrl_state = target_nonzero[index_differ]
    #         break

    target_cx = []
    for k in tg:
        if index_differ != k:
            if index_nonzero[k] != index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

    # target_cx = []
    # for index, value in enumerate(target_nonzero):
    #     if index_differ != index:
    #         if target_nonzero[index] != target_zero[index]:
    #             circuit.cx(target[index_differ], target[index], ctrl_state=ctrl_state)
    #             target_cx.append(index)

    target_remainder = []
    for k in rg:
        if index_nonzero[k] != index_zero[k]:
            circuit.cx(index_differ, k, ctrl_state=ctrl_state)
            target_remainder.append(k)

    # target_remainder = []
    # for index, value  in enumerate(remainder_nonzero):
    #     if remainder_nonzero[index] != remainder_zero[index]:
    #         circuit.cx(target[index_differ], remainder[index], ctrl_state=ctrl_state)
    #         target_remainder.append(index)

    tab = {'0':'1', '1':'0'}

    for k in rg:
        if index_zero[k] == '0':
            circuit.x(k)

    # for index, value in enumerate(remainder_zero):
    #     if value == '0':
    #         circuit.x(remainder[index])

    circuit.mcx(rg, index_differ)

    for k in rg:
        if index_zero[k] == '0':
            circuit.x(k)

    # for index, value in enumerate(remainder_zero):
    #     if value == '0':
    #         circuit.x(remainder[index])


    new_state = {}
    for index, value in state.items():
        if index == '1011':
            print(1)


        target_index = index[:n_qubits - target_size]
        remainder_index = index[n_qubits - target_size:]
        if target_index[index_differ] == ctrl_state:

            n_remainder_index = ''
            for k in range(len(remainder_index)):
                if k in target_remainder:
                    n_remainder_index = n_remainder_index + tab[remainder_index[k]]
                else:
                    n_remainder_index = n_remainder_index + remainder_index[k]

            n_target_index = ''
            for k in range(len(target_index)):
                if k in target_cx:
                    n_target_index = n_target_index + tab[target_index[k]]
                elif k == index_differ:
                    if n_remainder_index == remainder_nonzero:
                        n_target_index = n_target_index + tab[target_index[k]]
                    else:
                        n_target_index = n_target_index + target_index[k]
                else:
                    n_target_index = n_target_index + target_index[k]

            new_state[n_target_index[::-1] + n_remainder_index[::-1]] = state[index]
        else:
            new_state[index] = state[index]


    return circuit, new_state


def sparse_initialize(state):
    key, _ = list(state.items())[0]
    n_qubits = len(key)
    n_qubits = int(n_qubits)

    pivot_circuit = qiskit.QuantumCircuit(n_qubits)

    non_zero = len(state)

    target_size = np.log2(non_zero)
    target_size = np.ceil(target_size)
    target_size = int(target_size)
    next_state = state.copy()

    index_nonzero = _get_index_nz(next_state, n_qubits-target_size)

    while index_nonzero is not None:

        index_zero = _get_index_zero(n_qubits, non_zero, next_state)

        print('pivoting')
        circ, next_state = pivoting(index_zero, index_nonzero, target_size, next_state)
        pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)

        index_nonzero = _get_index_nz(next_state, target_size)

    dense_state = np.zeros(2**(target_size))
    for key, value in next_state.items():
        dense_state[int(key, 2)] = value
    initialize_circ = initialize(dense_state)
    sp_circuit = qiskit.QuantumCircuit(n_qubits)
    sp_circuit.compose(initialize_circ, sp_circuit.qubits[:target_size], inplace=True)
    sp_circuit.compose(pivot_circuit.reverse_ops(), sp_circuit.qubits, inplace=True)

    return sp_circuit


def _get_index_zero(n_qubits, non_zero, state):
    index_zero = None
    for k in range(2 ** non_zero):
        txt = '0' + str(n_qubits) + 'b'
        index = format(k, txt)
        if not index in state:
            index_zero = index
            break
    return index_zero


def _get_index_nz(state, target_size):
    index_nonzero = None
    for index, _ in state.items():
        if index[:target_size] != target_size * '0':
            index_nonzero = index
            break
    return index_nonzero


def _count_nonzero(state):
    non_zero = 0
    for k in state:
        if not np.isclose(k, 0.0):
            non_zero += 1
    return non_zero
