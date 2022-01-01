# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Functions to compute entanglement measures.
"""

import importlib.util
import numpy as np
import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from tensorly.decomposition import tucker

# pylint: disable=missing-function-docstring

def _get_iota(j: int, n: int, b: int, basis_state: int):
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


def meyer_wallach_entanglement(vector: np.ndarray):
    num_qb = _to_qubits(vector.shape[0])
    meyer_wallach_entry = np.zeros(shape=(num_qb, 1))
    for j in range(num_qb):
        psi_0 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=complex)
        psi_1 = np.zeros(shape=(vector.shape[0]//2, 1), dtype=complex)
        for basis_state, entry in enumerate(vector):
            delta_0, new_basis_state_0 = _get_iota(j, num_qb, 0, basis_state)
            delta_1, new_basis_state_1 = _get_iota(j, num_qb, 1, basis_state)

            if delta_0:
                psi_0[new_basis_state_0] = entry
            if delta_1:
                psi_1[new_basis_state_1] = entry

        entry = generalized_cross_product(psi_0, psi_1)
        meyer_wallach_entry[j] = entry

    return np.sum(meyer_wallach_entry) * (4/num_qb)


def geometric_entanglement(state_vector, return_product_state=False):
    tensorly_loader = importlib.util.find_spec('tensorly')
    if tensorly_loader is None:
        raise ImportError(
            "To calculate the geometric entanglement we use Tucker decomposition"
            " and use tensorly for that. Please install it, e.g., "
            "`pip install tensortly`."
        )

    shape = tuple([2] * _to_qubits(state_vector.shape[0]))
    rank = [1] * _to_qubits(state_vector.shape[0])
    tensor = tl.tensor(state_vector).reshape(shape)
    results = {}
    # The Tucker decomposition is actually a randomized algorithm.
    # We take three shots and take the min of it.
    for _ in range(3):
        decomp_tensor: TuckerTensor = tucker(
            tensor, rank=rank, verbose=False, svd='numpy_svd', init='random'
        )
        fidelity_loss = 1 - np.linalg.norm(decomp_tensor.core) ** 2
        results[fidelity_loss] = decomp_tensor

    min_fidelity_loss = min(results)

    if return_product_state:
        return min_fidelity_loss, decomp_tensor.factors

    return min_fidelity_loss


def _to_qubits(n_state_vector):
    return int(np.ceil(np.log2(n_state_vector))) if n_state_vector > 0 else 0
