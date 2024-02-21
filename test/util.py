""" auxiliary file """
from qiskit import transpile

def measurement(circuit, n_qubits, classical_register, backend, shots):
    """ run circuit and return measurements """
    circuit.measure(list(range(n_qubits)), classical_register)

    job = backend.run(
        transpile(circuit, backend),
        shots=shots,
        optimization_level=3
    )

    counts = job.result().get_counts(circuit)

    counts2 = {}
    for k in range(2 ** n_qubits):
        pattern = f'{k:0{n_qubits}b}'
        if pattern in counts:
            counts2[pattern] = counts[pattern]
        else:
            counts2[pattern] = 0.0

    return [value / shots for (key, value) in counts2.items()]
