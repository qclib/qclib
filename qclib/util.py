from qiskit import execute, Aer


def get_state(circ):
    backend = Aer.get_backend('statevector_simulator')
    state_vector = execute(circ, backend).result().get_statevector()
    
    return state_vector