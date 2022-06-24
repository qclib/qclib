from qiskit.extensions.quantum_initializer import UCGate as qUCGate
import numpy as np


class UCGate(qUCGate):
    """
    qiskit UCGate with fixed global phase
    FIXME: requires to extract the gate matrix
    """
    def __init__(self, gate_list, up_to_diagonal=False):
        super().__init__(gate_list, up_to_diagonal)

    def _define(self):
        super()._define()
        self.definition = super().definition

        import qiskit.quantum_info as qi

        op = qi.Operator(self.definition)
        actual_phase = np.angle(op.data[0, 0])
        original_phase = np.angle(self.params[0][0, 0])
        self.definition.global_phase = self.definition.global_phase + original_phase - actual_phase
