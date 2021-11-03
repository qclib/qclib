import logging

import numpy as np
import qutip

LOG = logging.getLogger(__name__)


def get_iota(j: int, n: int, b: int, basis_state: int):
    assert b in [0, 1]
    full_mask = 2**n - 1

    mask_j = 1 << j
    value = (mask_j & basis_state) >> j

    low_mask = full_mask >> (n - j)
    high_mask = full_mask & (full_mask << (j + 1))
    new_basis_state = ((basis_state & high_mask) >> 1) + (basis_state & low_mask)

    return value == b, new_basis_state


def generalized_cross_product(u: np.ndarray, v: np.ndarray):
    entries = []
    for j in range(u.shape[0]):
        for i in range(j):
            entry = np.abs(u[i] * v[j] - u[j] * v[i])**2
            entries.append(entry)
    return np.sum(entries)


def calculate_entropy_meyer_wallach(vector: np.ndarray):
    num_qb = int(np.ceil(np.log2(vector.shape[0])))
    meyer_wallach_entry = np.zeros(shape=(num_qb, 1))
    for j in range(num_qb):
        psi_0 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=np.complex)  # np.zeros(shape=())
        psi_1 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=np.complex)  # np.zeros(shape=())
        for basis_state, entry in enumerate(vector):
            delta_0, new_basis_state_0 = get_iota(j, num_qb, 0, basis_state)
            delta_1, new_basis_state_1 = get_iota(j, num_qb, 1, basis_state)

            if delta_0:
                psi_0[new_basis_state_0] = entry
            if delta_1:
                psi_1[new_basis_state_1] = entry

        entry = generalized_cross_product(psi_0, psi_1)
        meyer_wallach_entry[j] = entry

    return np.sum(meyer_wallach_entry) * (4/num_qb)


def compute_Q_ptrace(ket, N = None):
    """Computes Meyer-Wallach measure using alternative interpretation, i.e. as
    an average over the entanglements of each qubit with the rest of the system
    (see https://arxiv.org/pdf/quant-ph/0305094.pdf).

    COPIED from https://github.com/born-2learn/Entanglement_in_QML/blob/main/libraries/meyer_wallach_measure.py
    (APACHE 2.0 License)

    Args:
    =====
    ket : numpy.ndarray or list
        Vector of amplitudes in 2**N dimensions
    N : int
        Number of qubits

    Returns:
    ========
    Q : float
        Q value for input ket
    """
    if N is None:
        N = int(np.ceil(np.log2(ket.shape[0])))
    ket = qutip.Qobj(ket, dims=[[2] * (N), [1] * (N)]).unit()
    LOG.debug('KET=  ', ket)
    entanglement_sum = 0
    for k in range(N):
        LOG.debug('value of n', k, 'PTrace: ', ket.ptrace([k]) ** 2)
        rho_k_sq = ket.ptrace([k]) ** 2
        entanglement_sum += rho_k_sq.tr()

    Q = 2 * (1 - (1 / N) * entanglement_sum)
    return Q
