""" auxiliary file """
from qiskit import execute

def measurement(circuit, n_qubits, classical_register, backend, shots):
    """ run circuit and return measurements """
    circuit.measure(list(range(n_qubits)), classical_register)

    job = execute(circuit, backend, shots=shots, optimization_level=3)

    counts = job.result().get_counts(circuit)

    counts2 = {}
    for k in range(2 ** n_qubits):
        pattern = '{:0{}b}'.format(k, n_qubits)
        if pattern in counts:
            counts2[pattern] = counts[pattern]
        else:
            counts2[pattern] = 0.0

    return [value / shots for (key, value) in counts2.items()]
