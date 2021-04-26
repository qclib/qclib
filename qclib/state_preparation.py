"""
State preparation using Schmidt decomposition 	arXiv:1003.5760
"""
import numpy as np
import qiskit


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
    sp_circuit.unitary(u, B, 'u')

    # apply gate V to the second register
    sp_circuit.unitary(v.T, A, 'v')
    return sp_circuit
