import unittest
import numpy as np
from qclib.state_preparation.gleining import initialize
from qiskit import Aer, execute, QuantumCircuit


class TestGleining(unittest.TestCase): 

    def test_two_states_uniform(self):
        state = 1 / np.sqrt(2) * np.array([1, 0, 0, 0, 0, 1, 0, 0])
        circ = initialize(state)
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        self.assertTrue(np.allclose(result.get_statevector(), state))

    def test_three_states_superposition(self):
        state = 1 / np.sqrt(168) * np.array([0, 2, 0, 0, 8, 0, 0, 10])
        circ = initialize(state)
        backend = Aer.get_backend('statevector_simulator')
        job = execute(circ, backend)
        result = job.result()
        self.assertTrue(np.allclose(result.get_statevector(), state))
