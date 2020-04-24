from unittest import TestCase
import numpy as np
from qiskit import execute, Aer, QuantumRegister
from qclib.encoding import _recursive_compute_angles
from qclib import QuantumCircuit

class TestMultiplexor(TestCase):
    def test_ry_multiplexor(self):
        angles = []
        state = [np.sqrt(0.1), np.sqrt(0.2), np.sqrt(0.4), np.sqrt(0.3)]
        _recursive_compute_angles(state, angles)
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.ry(angles[0], qr[1])
        circuit.ry_multiplexor([angles[1], angles[2]], [qr[0], qr[1]])

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector(circuit)

        self.assertTrue(np.isclose(out_state, state).all())
