from qiskit.extensions.quantum_initializer import UCGate as qUCGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit


class UCGate(qUCGate):
    """
    qiskit UCGate with fixed inverse()
    """

    def __init__(self, gate_list, up_to_diagonal=False):
        super().__init__(gate_list, up_to_diagonal)

    def inverse(self):
        """Return the inverse.

        This does not re-compute the decomposition for the multiplexer with the inverse of the
        gates but simply inverts the existing decomposition.
        """
        inverse_gate = Gate(
            name=self.name + "_dg", num_qubits=self.num_qubits, params=[]
        )  # removing the params because arrays are deprecated

        definition = QuantumCircuit(*self.definition.qregs)
        for inst in reversed(self._definition):
            definition._append(inst.replace(operation=inst.operation.inverse()))

        definition.global_phase = -self.definition.global_phase

        inverse_gate.definition = definition
        return inverse_gate
