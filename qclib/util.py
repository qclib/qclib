from qiskit import execute, Aer
from qiskit.providers.aer import QasmSimulator


def get_counts(circ):

    backend = Aer.get_backend('qasm_simulator')
    counts = execute(circ, backend).result().get_counts()
    return counts


def get_state(circ):

    backend = Aer.get_backend('statevector_simulator')
    state_vector = execute(circ, backend).result().get_statevector()
    
    return state_vector
