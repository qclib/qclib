import numpy as np
import qiskit
from qclib.state_preparation.schmidt import initialize as dense_init

# pylint: disable=maybe-no-member

def initialize(state, aux=False):
    """ Create circuit to initialize a sparse quantum state arXiv:2006.00016

    For instance, to initialize the state a|001>+b|100>
        $ state = {'001': a, '100': b}
        $ circuit = sparse_initialize(state)

    Parameters
    ----------
    state: dict of {str:int}
        A unit vector representing a quantum state.
        Keys are binary strings and values are amplitudes.

    aux: bool
        circuit with auxiliary qubits if aux == True

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

    if aux:
        remain = list(range(n_qubits - target_size, n_qubits))
        n_anci = len(remain)

        memory = qiskit.QuantumRegister(n_qubits, name='q')
        anc = qiskit.QuantumRegister(n_anci-1, name='anc') #TODO
        pivot_circuit = qiskit.QuantumCircuit(anc, memory)

    else:
        memory = qiskit.QuantumRegister(n_qubits, name='q')
        if aux:
            pivot_circuit = qiskit.QuantumCircuit(memory, aux)
        else:
            pivot_circuit = qiskit.QuantumCircuit(memory)


    index_nonzero = _get_index_nz(next_state, n_qubits-target_size)

    while index_nonzero is not None:

        index_zero = _get_index_zero(n_qubits, non_zero, next_state)

        circ, next_state = _pivoting(index_zero, index_nonzero, target_size, next_state, aux)
        pivot_circuit.compose(circ, pivot_circuit.qubits, inplace=True)

        index_nonzero = _get_index_nz(next_state, n_qubits - target_size)

    dense_state = np.zeros(2**target_size)
    for key, value in next_state.items():
        dense_state[int(key, 2)] = value

    if non_zero <= 2:
        initialize_circ = qiskit.QuantumCircuit(1)
        initialize_circ.initialize(dense_state)
    else:
        initialize_circ = dense_init(dense_state)

    if aux==True:
        sp_circuit = qiskit.QuantumCircuit(anc, memory)
        nun_aux = n_anci-1  # TODO
        sp_circuit.compose(initialize_circ, sp_circuit.qubits[nun_aux:nun_aux+target_size], inplace=True)
        sp_circuit.barrier()
        sp_circuit.compose(pivot_circuit.reverse_bits().reverse_ops(), inplace=True)
    else:
        sp_circuit = qiskit.QuantumCircuit(n_qubits)
        sp_circuit.compose(initialize_circ, sp_circuit.qubits[:target_size], inplace=True)
        sp_circuit.compose(pivot_circuit.reverse_bits().reverse_ops(), sp_circuit.qubits, inplace=True)

    return sp_circuit


def _pivoting(index_zero, index_nonzero, target_size, state=None, aux=False):

    n_qubits = len(index_zero)
    target = list(range(n_qubits - target_size))
    remain = list(range(n_qubits - target_size, n_qubits))

    memory = qiskit.QuantumRegister(n_qubits)

    if aux:
        n_anci = len(remain)
        anc = qiskit.QuantumRegister(n_anci-1, name='anc')#TODO
        circuit = qiskit.QuantumCircuit(memory, anc)
    else:
        circuit = qiskit.QuantumCircuit(memory)

    index_differ = 0
    for k in target:
        if index_nonzero[k] != index_zero[k]:
            index_differ = k
            ctrl_state = index_nonzero[k]
            break

    target_cx = []
    for k in target:
        if index_differ != k and index_nonzero[k] != index_zero[k]:
            circuit.cx(index_differ, k, ctrl_state=ctrl_state)
            target_cx.append(k)

    for k in remain:
        if index_nonzero[k] != index_zero[k]:
            circuit.cx(index_differ, k, ctrl_state=ctrl_state)
            target_cx.append(k)



    for k in remain:
        if index_zero[k] == '0':
            circuit.x(k)

    if aux == True:
        # apply mcx using mode v-chain
        mcxvchain(circuit, memory, anc, remain, index_differ)
    else:
        circuit.mcx(remain, index_differ)

    for k in remain:
        if index_zero[k] == '0':
            circuit.x(k)

    new_state = _next_state(ctrl_state, index_differ, index_zero, remain, state, target_cx)

    return circuit, new_state


def _next_state(ctrl_state, index_differ, index_zero, remain, state, target_cx):
    tab = {'0': '1', '1': '0'}
    new_state = {}
    for index, _ in state.items():

        if index[index_differ] == ctrl_state:

            n_index = ''
            for k, _ in enumerate(index):
                if k in target_cx:
                    n_index = n_index + tab[index[k]]
                else:
                    n_index = n_index + index[k]

        else:
            n_index = index

        if n_index[remain[0]:] == index_zero[remain[0]:]:
            n_index = n_index[:index_differ] + tab[index[index_differ]] + n_index[index_differ + 1:]

        new_state[n_index] = state[index]
    return new_state

def _get_index_zero(n_qubits, non_zero, state):
    index_zero = None
    for k in range(2 ** non_zero):
        txt = '0' + str(n_qubits) + 'b'
        index = format(k, txt)

        if index not in state:
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


def mcxvchain(circuit, memory, anc, lst_ctrl, tgt):
    circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
    for j in range(2, len(lst_ctrl)):
        circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])

    circuit.cx(anc[len(lst_ctrl) - 2], tgt)

    for j in reversed(range(2, len(lst_ctrl))):
        circuit.rccx(memory[lst_ctrl[j]], anc[j - 2], anc[j - 1])
    circuit.rccx(memory[lst_ctrl[0]], memory[lst_ctrl[1]], anc[0])
