'''
Quantum Multiplexor
'''

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Instruction
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate


class Multiplexor(Instruction):
    """
    Rotation R_k Multiplexor
    """
    def __init__(self, params, gate):
        """
        Create multiplexor
        params (list): angles of the multiplexor
        """
        self._gate = gate
        self.params = params
        self.num_qubits = int(np.log2(len(params)) + 1)
        self._circuit = None
        self.cnot_control = None
        self.multiplexor_angles = None

        super().__init__("multiplexor", self.num_qubits, 0, params)

    def _define(self):

        self._compute_cnot_control()
        self._compute_multiplexor_angles()
        self._mirror_angles()
        self._initialize()

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
        self.multiplexor_angles = self.params.copy()
        left = 0
        right = len(self.params)
        mid = (left + right) // 2
        self._compute_multiplexor_angles2(left, mid, right)

    def _compute_multiplexor_angles2(self, left, mid, right):
        for k in range(0, mid-left):
            v_left = (self.multiplexor_angles[k+left] + self.multiplexor_angles[k+mid]) / 2
            v_right = (self.multiplexor_angles[k+left] - self.multiplexor_angles[k+mid]) / 2
            self.multiplexor_angles[k+left] = v_left
            self.multiplexor_angles[k+mid] = v_right

        if mid - left > 1:
            mid_right = (mid + right) // 2
            mid_left = (left + mid) // 2

            self._compute_multiplexor_angles2(left, mid_left, mid)
            self._compute_multiplexor_angles2(mid, mid_right, right)

    def _mirror_angles(self):
        n_angles = len(self.multiplexor_angles)
        if n_angles >= 4:
            j = 2
            while j <= n_angles // 2:
                for k in range(j, n_angles, 2*j):
                    self.multiplexor_angles[k:k+j] = reversed(self.multiplexor_angles[k:k+j])
                j = j * 2

    def _initialize(self):
        quantum_register = QuantumRegister(self.num_qubits)
        self._circuit = QuantumCircuit(quantum_register)

        m_angles = self.multiplexor_angles
        for k, _ in enumerate(self.params):
            self._circuit.append(self._gate(m_angles[k]), [quantum_register[0]])
            if self.cnot_control[k] != 0:
                self._circuit.cx(quantum_register[self.cnot_control[k]], quantum_register[0])

        self._circuit.cx(quantum_register[self.num_qubits - 1], quantum_register[0])

    def ry_multiplexor(self, params, qubits):
        """
        Apply Ry multiplexor with angles params arXiv:quant-ph/0406176
        """
        return self.append(Multiplexor(params, RYGate), qubits)

    def rz_multiplexor(self, params, qubits):
        """
        Apply Ry multiplexor with angles params arXiv:quant-ph/0406176
        """
        return self.append(Multiplexor(params, RZGate), qubits)

    QuantumCircuit.ry_multiplexor = ry_multiplexor
    QuantumCircuit.rz_multiplexor = rz_multiplexor
