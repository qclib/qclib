from unittest import TestCase
from scipy.stats import unitary_group
from qclib.unitary import unitary, _compute_gates, _qsd
from qclib.util import get_state
import numpy as np
from scipy.linalg import block_diag

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

    def test_unitary_qsd_3qubits(self):
        u = unitary_group.rvs(16)
        gate = unitary(u, 'qsd')
        state = get_state(gate)
        self.assertTrue(np.allclose(u[:, 0], state))

    def test_compute_gates(self):
        gate1 = unitary_group.rvs(8)
        gate2 = unitary_group.rvs(8)

        d, V, W = _compute_gates(gate1, gate2)

        calc1 = V @ np.diag(d) @ W
        calc2 = V @ np.diag(d).conj().T @ W
        self.assertTrue(np.allclose(calc1, gate1))
        self.assertTrue(np.allclose(calc2, gate2))

    def test_qsd(self):
        gate1 = unitary_group.rvs(2)
        gate2 = unitary_group.rvs(2)

        d, V, W = compute_gates(gate1, gate2)

        gate = _qsd(gate1, gate2)

        state = get_state(gate)
        print(state)




