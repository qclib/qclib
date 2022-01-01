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
from typing import Union, Tuple, List

import numpy as np
import tensorly as tl
from tensorly.tucker_tensor import TuckerTensor
from tensorly.decomposition import tucker


def _get_iota(qubit_idx: int, qubits: int, selector_bit: int, basis_state: int):
    assert selector_bit in [0, 1]
    full_mask = 2**qubits - 1

    mask_j = 1 << qubit_idx
    value = (mask_j & basis_state) >> qubit_idx

    low_mask = full_mask >> (qubits - qubit_idx)
    high_mask = full_mask & (full_mask << (qubit_idx + 1))
    new_basis_state = ((basis_state & high_mask) >> 1) + (basis_state & low_mask)

    return value == selector_bit, new_basis_state


def generalized_cross_product(vector_u: np.ndarray, vector_v: np.ndarray) -> np.ndarray:
    """
    Calculates the generalized cross product (see Eqn. (3) in quant-ph/0305094)

    Args:
        vector_u (array-like):
            The first vector (called u)
        vector_v (array-like):
            The first vector (called u)

    Returns:
        array-like: the resulting vector

    """
    entries = []
    for j in range(vector_u.shape[0]):
        for i in range(j):
            entry = np.abs(vector_u[i] * vector_v[j] - vector_u[j] * vector_v[i])**2
            entries.append(entry)
    return np.sum(entries)


def meyer_wallach_entanglement(vector: np.ndarray) -> float:
    """

    Computes the Meyer-Wallach entanglement (1,2) of a quantum state.

    [1] Meyer, D. A. & Wallach, N. R. Global entanglement in multiparticle systems. J Math Phys 43,
        4273–4278 (2002).
    [2] Brennen, G. An observable measure of entanglement for pure states of multi-qubit systems.
        P Soc Photo-opt Ins 3, 619–626 (2003).

    Args:
        vector (array-like):
            The vector of the quantum state (in computational basis)
    Returns:
        float: the entanglement which is between 0 and 1 (highest is 1)

    """
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


def geometric_entanglement(state_vector: np.ndarray, return_product_state=False
                           ) -> Union[float, Tuple[float, List[np.ndarray]]]:
    """

    Computes the geometric entanglement (1,2) of a quantum state.

    [1] SHIMONY, A. Degree of Entanglementa. Ann Ny Acad Sci 755, 675–679 (1995).

    [2] Barnum, H. & Linden, N. Monotones and invariants for multi-particle quantum states.
        J Phys Math Gen 34, 6787 (2001).

    Args:
        state_vector (array-like):
            The vector of the quantum state (in computational basis)

        return_product_state (bool):
            If True, return the list of product states too.

    Returns:
        float or Tuple[float, List[array-like]]: #
            the entanglement which is between 0 and 1 (highest is 1).
            If return_product_state == True, returns a tuple with a list of product state vectors.

    """
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
        return min_fidelity_loss, [f.flatten() for f in results[min_fidelity_loss].factors]

    return min_fidelity_loss


def _to_qubits(n_state_vector):
    return int(np.ceil(np.log2(n_state_vector))) if n_state_vector > 0 else 0
