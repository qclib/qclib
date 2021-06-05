from unittest import TestCase
from scipy.stats import unitary_group
from qclib.unitary import unitary
from qclib.util import get_state
import numpy as np

class TestUnitary(TestCase):
    def test_unitary_csd_2qubits(self):
        u = unitary_group.rvs(4)
        gate = unitary(u)
        state = get_state(gate)

        self.assertTrue(np.allclose(u[:, 0], state))

    def test_unitary_csd_5qubits(self):
        u = unitary_group.rvs(32)
        gate = unitary(u)
        state = get_state(gate)
        self.assertTrue(np.allclose(u[:, 0], state))