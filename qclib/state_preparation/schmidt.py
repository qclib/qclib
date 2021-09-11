import numpy as np
import qiskit
from qclib.unitary import unitary

# pylint: disable=maybe-no-member


def initialize(state, rank=0):
    """ State preparation using Schmidt decomposition arXiv:1003.5760

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize(state)

    Parameters
    ----------
    state: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    rank: int
        ``state`` low-rank approximation (1 <= ``rank`` < 2**(n_qubits//2)).
        If ``rank`` is not in the valid range, it will be ignored and the full
        ``rank`` will be used.

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """
    
    s = np.copy(state)

    size = len(s)
    n_qubits = np.log2(size)

    r = n_qubits % 2

    s.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + r)))

    u, d, v = np.linalg.svd(s)
    if (rank >= 1 and rank <= min(s.shape)):
        """
        # This commented out code is equivalent to the one below.
        # It is more compact, but the one below is easier to read.
        d = d[:rank]
        u = (u.T[:rank]).T
        v = v[:rank]
        s = u @ np.diag(d) @ v
        """
        s = np.zeros((len(u), len(v)), dtype=complex)
        for i in range(rank):
            s += d[i] * np.outer(u.T[i], v[i])
               
        u, d, v = np.linalg.svd(s)
        
    d = d / np.linalg.norm(d)

    A = qiskit.QuantumRegister(n_qubits//2 + r)
    B = qiskit.QuantumRegister(n_qubits//2)

    circuit = qiskit.QuantumCircuit(A, B)

    circuit.initialize(d, B)

    for k in range(int(n_qubits//2)):
        circuit.cx(B[k], A[k])

    # apply gate U to the first register
    gate_u = unitary(u, 'qsd')
    circuit.compose(gate_u, B, inplace=True)

    # apply gate V to the second register
    gate_v = unitary(v.T, 'qsd')
    circuit.compose(gate_v, A, inplace=True)

    return circuit
