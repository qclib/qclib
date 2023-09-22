""" auxiliary file """
from qiskit import execute

def measurement(circuit, n_qubits, classical_register, backend, shots, dense=True, patterns=None):
    """ run circuit and return measurements """
    circuit.measure(list(range(n_qubits)), classical_register)

    job = execute(circuit, backend, shots=shots, optimization_level=3)

    counts = job.result().get_counts(circuit)

    if dense:
        counts2 = {}
        for k in range(2 ** n_qubits):
            pattern = f'{k:0{n_qubits}b}'
            if pattern in counts:
                counts2[pattern] = counts[pattern]
            else:
                counts2[pattern] = 0
        counts = counts2
    else:
        if patterns:
            for pattern in patterns:
                if not pattern in counts:
                    counts[pattern] = 0
        counts = {i: counts[i] for i in sorted(counts)}

    return [value / shots for (_, value) in counts.items()]
