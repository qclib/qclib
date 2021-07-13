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
    circuit = qiskit.QuantumCircuit(n_qubits)

    tg = list(range(n_qubits - target_size))
    rg = list(range(n_qubits - target_size, n_qubits))

    index_differ = 0
    for k in tg:
        if index_nonzero[k] != index_zero[k]:
            index_differ = k
            ctrl_state = index_nonzero[k]
            break

    target_cx = []
    for k in tg:
        if index_differ != k:
            if index_nonzero[k] != index_zero[k]:
                circuit.cx(index_differ, k, ctrl_state=ctrl_state)
                target_cx.append(k)

    for k in rg:
        if index_nonzero[k] != index_zero[k]:
            circuit.cx(index_differ, k, ctrl_state=ctrl_state)
            target_cx.append(k)

    tab = {'0':'1', '1':'0'}

    for k in rg:
        if index_zero[k] == '0':
            circuit.x(k)

    circuit.mcx(rg, index_differ)

    for k in rg:
        if index_zero[k] == '0':
            circuit.x(k)

    new_state = {}
    for index, value in state.items():

        if index[index_differ] == ctrl_state:

            n_index = ''
            for k, value in enumerate(index):
                if k in target_cx:
                    n_index = n_index + tab[index[k]]
                else:
                    n_index = n_index + index[k]


        else:
            n_index = index
        if n_index[rg[0]:] == index_zero[rg[0]:]:
            n_index = n_index[:index_differ] + tab[index[index_differ]] + n_index[index_differ+1:]

        new_state[n_index] = state[index]


    return circuit, new_state


def sparse_initialize(state):
    """ Create circuit to initialize a sparse quantum state arXiv:2006.00016

    For instance, to initialize the state a|001>+b|100>
        $ state = {'001': a, '100': b}
        $ circuit = sparse_initialize(state)

    Parameters
    ----------
    state: dict of {str:int}
        A unit vector representing a quantum state. Keys are binary strings and values are amplitudes.

    Returns
    -------
    sp_circuit: QuantumCircuit
        QuantumCircuit to initialize the state
    """
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

        circ, next_state = pivoting(index_zero, index_nonzero, target_size, next_state)
        pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)

        index_nonzero = _get_index_nz(next_state, n_qubits - target_size)

    dense_state = np.zeros(2**(target_size))
    for key, value in next_state.items():

        dense_state[int(key, 2)] = value
    if non_zero <= 2:
        initialize_circ = qiskit.QuantumCircuit(1)
        initialize_circ.initialize(dense_state)
    else:
        initialize_circ = initialize(dense_state)
    sp_circuit = qiskit.QuantumCircuit(n_qubits)
    sp_circuit.compose(initialize_circ, sp_circuit.qubits[:target_size], inplace=True)
    sp_circuit.barrier()
    sp_circuit.compose(pivot_circuit.reverse_bits().reverse_ops(), sp_circuit.qubits, inplace=True)

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
        if index[:target_size] != (target_size) * '0':
            index_nonzero = index
            break
    return index_nonzero


def _count_nonzero(state):
    non_zero = 0
    for k in state:
        if not np.isclose(k, 0.0):
            non_zero += 1
    return non_zero
