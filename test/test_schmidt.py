from unittest import TestCase
import numpy as np
from qclib.state_preparation.schmidt import initialize
from qclib.util import get_state
from qiskit import transpile


class TestInitialize(TestCase):

    @staticmethod
    def mae(state, ideal):
        """
         Mean Absolute Error
        """
        return np.sum(np.abs(state-ideal))/len(ideal)
        
    def test_initialize(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a)

        state = get_state(circuit)

        self.assertTrue(np.allclose(a, state))
    
    def test_initialize_low_rank(self):
        a = np.random.rand(32) + np.random.rand(32) * 1j
        a = a / np.linalg.norm(a)

        circuit = initialize(a, rank=3)

        state = get_state(circuit)

        print(TestInitialize.mae(state,a))

        self.assertTrue(TestInitialize.mae(state,a) < 0.04)
