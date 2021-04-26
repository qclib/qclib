from unittest import TestCase
import numpy as np
from qclib.state_preparation import initialize
import qiskit

class TestInitialize(TestCase):
    def test_initialize(self):
        a = np.random.rand(16) + np.random.rand(16) * 1j
        a = a / np.linalg.norm(a)

        circ = initialize(a)

        state = get_state(circ1)

        print(np.allclose(a, state))
