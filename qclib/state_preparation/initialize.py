from qiskit.circuit.gate import Gate
import numpy as np


class Initialize(Gate):

    @staticmethod
    def initialize(q_circuit, state, qubits):
        pass

    def _get_num_qubits(self, params):
        self.num_qubits = np.log2(len(params))
        if not self.num_qubits.is_integer():
            Exception("The number of amplitudes is not a power of 2")
        self.num_qubits = int(self.num_qubits)

    def validate_parameter(self, parameter):
        if isinstance(parameter, (int, float, complex)):
            return complex(parameter)
        elif isinstance(parameter, np.number):
            return complex(parameter.item())
        else:
            raise Exception(f"invalid param type {type(parameter)} for instruction  {self.name}")

