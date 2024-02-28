""" auxiliary file """
# from qiskit import execute
#
#
# def measurement(circuit, n_qubits, classical_register, backend, shots):
#     """ run circuit and return measurements """
#     circuit.measure(list(range(n_qubits)), classical_register)
#
#     job = execute(circuit, backend, shots=shots, optimization_level=3)
#
#     counts = job.result().get_counts(circuit)
#
#     count_s2 = {}
#     for k in range(2 ** n_qubits):
#         pattern = f'{k:0{n_qubits}b}'
#         if pattern in counts:
#             count_s2[pattern] = counts[pattern]
#         else:
#             count_s2[pattern] = 0.0
#
#     return [value / shots for (key, value) in count_s2.items()]

