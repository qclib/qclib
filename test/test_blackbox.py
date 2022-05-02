from unittest import TestCase
from qclib.state_preparation.blackbox import initialize
from qclib.util import get_state
import numpy as np


class TestBlackbox(TestCase):
    def test_blackbox(self):
        state = np.random.rand(16) - 0.5 + (np.random.rand(16) - 0.5) * 1j
        state = state / np.linalg.norm(state)

        qc = initialize(state)
        out = get_state(qc)

        out = out.reshape((len(out)//2, 2))
        out = out[:, 0]

        self.assertTrue(np.allclose(state, out, atol=0.02))
