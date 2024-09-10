from unittest import TestCase

import numpy as np
from scipy.stats import unitary_group, ortho_group
from qclib.sparse_isometry import householder_reflection_zero, generalized_householder_reflection, householder_decomposition
from qiskit.quantum_info import Operator
from qiskit import transpile


def householder_reflection_matrix(U):
    """
    U: 4 X 4 unitary matrix
    """

    a = U[:, [0]]
    b = np.zeros((4, 1), complex)
    b[0] = 1
    z = a - b
    M = (z @ z.conj().T) / (z.T @ z.conj())
    y = -(z.conj().T @ b) / (z.conj().T @ a)
    H = np.eye(4, dtype=complex) - (1 + y) * M
    return H


class TestSparseIso(TestCase):
    def test_householder_reflection_zero_real(self):
        h0 = householder_reflection_zero(2)
        mh0 = np.eye(4)
        mh0[0, 0] = -1
        self.assertTrue(np.allclose(Operator(h0).data, mh0))

    def test_householder_reflection_zero_complex(self):
        phi = 0.3
        h0 = householder_reflection_zero(2, phi)
        mh0 = np.eye(4, dtype=complex)
        mh0[0, 0] = np.e ** (1j * phi)
        m = Operator(h0).data
        self.assertTrue(np.allclose(Operator(h0).data, mh0))

    def test_householder_reflection_real(self):
        """
        Test householder_reflection
        """
        np.random.seed(1)
        matriz1 = ortho_group.rvs(4)

        matrix_H = householder_reflection_matrix(matriz1)
        x = matriz1[: , 0]
        y = [1., 0., 0., 0.]

        # circ = generalized_householder_reflection(2, matriz1[: , 0], [1, 0, 0, 0])
        circ = generalized_householder_reflection(y-x, x, y)
        mcirc = Operator(circ).data
        self.assertTrue(np.allclose(mcirc, matrix_H))

    def test_householder_reflection_complex(self):
        """
        Test householder_reflection
        """
        np.random.seed(1)
        matriz1 = unitary_group.rvs(4)

        matrix_H = householder_reflection_matrix(matriz1)
        x = matriz1[: , 0]
        y = [1, 0, 0, 0]

        circ = generalized_householder_reflection(y-x, x, y)
        mcirc = Operator(circ).data
        self.assertTrue(np.allclose(mcirc, matrix_H))

    def test_householder_decomposition_real(self):
        """
        Test householder_decomposition
        """
        np.random.seed(1)
        matriz1 = ortho_group.rvs(4)
        matriz1 = matriz1[:, [0, 1]]

        circ = householder_decomposition(matriz1)
        mcirc = Operator(circ).data
        self.assertTrue(np.allclose(matriz1, mcirc.T[:, [0, 1]]))

    def test_householder_decomposition_complex(self):
        """
        Test householder_decomposition
        """
        np.random.seed(1)
        matriz1 = unitary_group.rvs(16)
        matriz1 = matriz1[:, [0, 1, 2]]

        circ = householder_decomposition(matriz1)
        mcirc = Operator(circ).data
        self.assertTrue(np.allclose(matriz1, mcirc.conj().T[:, [0, 1, 2]]))

    def test_householder_decomposition_complex(self):
        """
        Test householder_decomposition
        """
        np.random.seed(1)
        matriz1 = unitary_group.rvs(4)
        matriz1 = matriz1[:, [0, 1]]

        circ = householder_decomposition(matriz1)
        t_circ = transpile(circ, basis_gates=['u', 'cx'])
        print(t_circ.count_ops())