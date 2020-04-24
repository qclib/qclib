import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate


class Multiplexor(Instruction):
    """
    Rotation R_k Multiplexor arXiv:quant-ph/0406176
    """
    def __init__(self, params, gate):
        """
        Create multiplexor
        params (list): angles of the multiplexor
        """
        self._gate = gate
        self.params = params
        self.num_qubits = int(np.log2(len(params)) + 1)

        super().__init__("multiplexor", self.num_qubits, 0, params)

    def _define(self):
        self._compute_cnot_control()
        self.initialize()
        self.definition = self._circuit.data

    def _compute_cnot_control(self):

        self.cnot_control = len(self.params) * [0]
        control_qubit = int(self.num_qubits - 1)
        step = len(self.params)
        pos = len(self.params) // 2

        while control_qubit >= 1:
            for j in range(pos-1, len(self.params), step):
                self.cnot_control[j] = control_qubit
            control_qubit = control_qubit - 1
            step = step // 2
            pos = pos // 2

    def _compute_multiplexor_angles(self):

        local_angles = self.params
        multiplexor_angles = len(local_angles) * [0]

        weights1 = np.array([(-1) ** k for k in range(len(local_angles))])
        weights2 = np.array([(-1) ** ((2 & (k + 1)) // 2) for k in range(len(local_angles))])

        for k, _ in enumerate(local_angles):
            if k != 0 and k % 2 == 0:
                local_angles = np.multiply(local_angles, weights2)
            if k % 2 == 1:
                local_angles = np.multiply(local_angles, weights1)
            multiplexor_angles[k] = sum(local_angles)/len(local_angles)

        end_multiplexor_angles = multiplexor_angles[:len(self.params) // 2]
        multiplexor_angles.reverse()
        end_multiplexor_angles = end_multiplexor_angles + multiplexor_angles[:len(self.params) // 2]
        return end_multiplexor_angles

    def initialize(self):
        qr = QuantumRegister(self.num_qubits)
        self._circuit = QuantumCircuit(qr)

        m_angles = self._compute_multiplexor_angles()

        for k, _ in enumerate(self.params):
            self._circuit.append(self._gate(m_angles[k]), [qr[0]])
            if self.cnot_control[k] != 0:
                self._circuit.cx(qr[self.cnot_control[k]], qr[0])

        self._circuit.cx(qr[self.num_qubits - 1], qr[0])

    def ry_multiplexor(self, params, qubits):
        return self.append(Multiplexor(params, RYGate), qubits)

    def rz_multiplexor(self, params, qubits):
        return self.append(Multiplexor(params, RZGate), qubits)

    QuantumCircuit.ry_multiplexor = ry_multiplexor
    QuantumCircuit.rz_multiplexor = rz_multiplexor
