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
Bounded Approximation Algorithm.
https://arxiv.org/abs/2111.03132
"""

from dataclasses import dataclass
from itertools import combinations, chain
from typing import List, Union, Tuple

import numba
import numpy as np
import tensorly as tl
from qclib.state_preparation.schmidt import cnot_count as schmidt_cnots

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring


def adaptive_approximation(state_vector, max_fidelity_loss,
                            strategy='greedy', max_combination_size=0, use_low_rank=False):
    """
    It reduces the entanglement of the given state, producing an approximation
    to reduce the complexity of the quantum circuit needed to prepare it.
    `https://arxiv.org/abs/2111.03132`_.
    Args:
        state_vector (list):
            A state vector to be approximated by a less complex state.
        max_fidelity_loss (float):
            Maximum fidelity loss allowed to the approximated state.
        strategy (string):
            Method to search for the best approximation ('brute_force' or 'greedy').
            For states larger than 2**8, the greedy strategy should preferably be used.
            Default is strategy='greedy'.
        max_combination_size (int):
            Maximum size of the combination ``C(n_qubits, max_combination_size)``
            between the qubits of an entangled subsystem of length ``n_qubits`` to
            produce the possible bipartitions
            (1 <= ``max_combination_size`` <= ``n_qubits``//2).
            For example, if ``max_combination_size``==1, there will be ``n_qubits``
            bipartitions between 1 and ``n_qubits``-1 qubits.
            The default value is 0 (the size will be maximum for each level).
        use_low_rank (bool):
            If set to True, non-rank-1 approximations are also considered. This is fine tuning for high-entanglement
            states and is slower.
            The default value is False

    Returns:
        Node: a node with the data required to build the quantum circuit.
    """
    state_vector = np.asarray(state_vector).reshape(-1, 1)
    n_qubits = _to_qubits(state_vector.shape[0])
    qubits = [list(range(n_qubits))]
    vectors = [state_vector]

    entanglement, product_state = geometric_entanglement(state_vector, return_product_state=True)
    if max_fidelity_loss >= entanglement:
        full_cnots = schmidt_cnots(state_vector)
        qubits = [[n] for n in range(len(product_state))]
        ranks = [1 for _ in range(len(product_state))]
        return Node(full_cnots, full_cnots, entanglement, entanglement, product_state, qubits, ranks, [])

    root_node = Node(0, 0, 0.0, 0.0, vectors, qubits, [0], [])
    _build_approximation_tree(root_node, max_fidelity_loss, strategy, max_combination_size, use_low_rank)

    leafs = []
    _search_leafs(root_node, leafs)

    best_node = _search_best(leafs)

    return best_node


def geometric_entanglement(state_vector, return_product_state=False):
    # node = adaptive_approximation(state_vector, 10.0, strategy='greedy', max_combination_size=1, use_low_rank=False)
    # return node.total_fidelity_loss
    from tensorly.tucker_tensor import TuckerTensor
    from tensorly.decomposition import tucker

    shape = tuple([2] * _to_qubits(state_vector.shape[0]))
    rank = [1] * _to_qubits(state_vector.shape[0])
    tensor = tl.tensor(state_vector).reshape(shape)
    results = {}
    # The Tucker decomposition is actually a randomized algorithm. We take three shots and take the min of it.
    for _ in range(3):
        decomp_tensor: TuckerTensor = tucker(tensor, rank=rank, verbose=False, svd='numpy_svd', init='random')
        fidelity_loss = 1 - np.linalg.norm(decomp_tensor.core) ** 2
        results[fidelity_loss] = decomp_tensor

    min_fidelity_loss = min(results)

    if return_product_state:
        return min_fidelity_loss, decomp_tensor.factors
    else:
        return min_fidelity_loss

@dataclass
class Node:
    """
    Tree node used in _approximation_tree function
    """
    node_saved_cnots: int
    total_saved_cnots: int

    node_fidelity_loss: float
    total_fidelity_loss: float

    vectors: List[Union[List[complex], np.ndarray]]
    qubits: List[List[int]]
    k_approximation: List[int]

    nodes: List['Node']

    def __str__(self):
        return f'total_saved_cnots:{self.total_saved_cnots}\n' + \
               f'total_fidelity_loss:{self.total_fidelity_loss}\n' + \
               f'len(subsystems):{len(self.qubits)}'


def _build_approximation_tree(node, max_fidelity_loss, strategy='brute_force', max_k=0, use_low_rank=False):
    # Ignore the completely disentangled qubits.
    data = [(q, k, v) for q, v, k in zip(node.qubits, node.vectors, node.k_approximation) if k == 0]

    for entangled_qubits, entangled_rank, entangled_vector in data:

        if not 1 <= max_k <= len(entangled_qubits)//2:
            max_k = len(entangled_qubits)//2

        if strategy == 'greedy':
            combs = _greedy_combinations(entangled_vector, entangled_qubits, max_k, max_fidelity_loss, use_low_rank)
        else:
            combs = _all_combinations(entangled_qubits, max_k)

        # Disentangles each bipartion from entangled_qubits combinations.
        for register_to_disentangle in combs:
            # Computes the two state vectors after disentangling "register_to_disentangle".
            possible_fidelity_loss = max(max_fidelity_loss - node.total_fidelity_loss, 0)
            node_fidelity_loss, subsystem1, subsystem2 = _compute_schmidt(
                entangled_vector, entangled_qubits, register_to_disentangle, possible_fidelity_loss, use_low_rank
            )
            total_fidelity_loss = 1 - (1 - node_fidelity_loss) * (1 - node.total_fidelity_loss)

            # Recursion should not continue in this branch if "total_fidelity_loss" has
            # reached "max_fidelity_loss". The leaf corresponds to the node of best
            # approximation of "max_fidelity_loss" on the branch.
            if total_fidelity_loss <= max_fidelity_loss:
                index = node.qubits.index(entangled_qubits)
                new_node = _create_node(node, index, register_to_disentangle,
                                            node_fidelity_loss, subsystem1, subsystem2)
                # Create one node for each bipartition.
                node.nodes.append(new_node)
                del new_node

    if len(node.nodes) > 0:  # If it is not the end of the recursion,
        node.vectors.clear() # clear vectors and qubits to save memory.
        node.qubits.clear()  # This information is no longer needed from this point
                             # on (but may be needed in the future).
    if strategy == 'greedy' and len(node.nodes) > 0:
        # Locally optimal choice at each stage.
        node.nodes = [_search_best(node.nodes)]

    for new_node in node.nodes:
        # call _build_approximation_tree recurrently for each new node.
        # except that the vectors are matrices. In this case we are done.
        if any(np.asarray(new_node.k_approximation) < 1):
            _build_approximation_tree(new_node, max_fidelity_loss, strategy, max_k, use_low_rank=use_low_rank)


def _all_combinations(entangled_qubits, max_k):
    return chain.from_iterable(combinations(entangled_qubits, k) for k in range(1, max_k+1))


def _greedy_combinations(entangled_vector, entangled_qubits, max_k, max_fidelity_loss, use_low_rank):
    """
    Combinations with a qubit-by-qubit analysis.
    Returns only one representative of the bipartitions of size k (1<=k<=max_k).
    The increment in the partition size is done by choosing the qubit that has
    the lowest fidelity-loss when removed from the remaining entangled subsystem.
    """
    node = Node(0, 0, 0.0, 0.0, [entangled_vector], [entangled_qubits, []], [0], [])
    best_nodes = []
    for _ in range(max_k):
        left_qubits = node.qubits[0]
        right_qubits = node.qubits[1] if len(node.qubits) > 1 else []

        nodes = []
        # Disentangles one qubit at a time.
        for qubit_to_disentangle in left_qubits:
            disentangled_qubits = right_qubits + [qubit_to_disentangle]
            node_fidelity_loss, subsystem1, subsystem2 = \
                _compute_schmidt(entangled_vector, entangled_qubits, disentangled_qubits, max_fidelity_loss, use_low_rank)
            cnots = _count_saved_cnots(entangled_vector, subsystem1, subsystem2)
            new_node = Node(
                cnots, cnots, node_fidelity_loss, node_fidelity_loss,
                [entangled_vector], [list(set(entangled_qubits) - set(disentangled_qubits))] + [disentangled_qubits],
                [0], []
            )
            nodes.append(new_node)
        # Search for the node with lowest fidelity-loss.
        node = _search_best(nodes)
        best_nodes.append(node)

    combs = [tuple(n.qubits[-1]) for n in best_nodes]
    return combs


def _create_node(node, index, qubits_to_disentangle, node_fidelity_loss, subsystem1_vector, subsystem2_vector):
    total_fidelity_loss = 1 - (1 - node_fidelity_loss) * (1 - node.total_fidelity_loss)

    vectors = node.vectors.copy()
    qubits = node.qubits.copy()
    k_approximation = node.k_approximation.copy()

    entangled_vector = vectors.pop(index)
    entangled_qubits = qubits.pop(index)
    k_approximation.pop(index)

    subsystem1_qubits = list(set(entangled_qubits).difference(set(qubits_to_disentangle)))
    subsystem2_qubits = qubits_to_disentangle

    node_saved_cnots = \
        _count_saved_cnots(entangled_vector, subsystem1_vector, subsystem2_vector)

    total_saved_cnots = node.total_saved_cnots + node_saved_cnots

    subsystem1_vector_rank = subsystem1_vector.shape[1] if len(subsystem1_vector.shape) == 2 else 1
    subsystem2_vector_rank = subsystem2_vector.shape[1] if len(subsystem2_vector.shape) == 2 else 1

    if subsystem1_vector_rank == subsystem2_vector_rank == 1:
        vectors.append(subsystem1_vector)
        qubits.append(subsystem1_qubits)
        subsystem1_k_approximation = 1 if subsystem1_vector.shape[0] == 2 else 0
        k_approximation.append(subsystem1_k_approximation)

        vectors.append(subsystem2_vector)
        qubits.append(subsystem2_qubits)
        subsystem2_k_approximation = 1 if subsystem2_vector.shape[0] == 2 else 0
        k_approximation.append(subsystem2_k_approximation)

    else:
        vectors.append(entangled_vector)
        qubits.append(entangled_qubits)
        k_approximation.append(subsystem1_vector.shape[1])

    return Node(
        node_saved_cnots, total_saved_cnots, node_fidelity_loss, total_fidelity_loss, vectors, qubits,
        k_approximation, []
    )


def _search_leafs(node, leafs):
    # It returns the leaves of the tree. These nodes are the ones with
    # total_fidelity_loss closest to max_fidelity_loss for each branch.
    if len(node.nodes) == 0:
        leafs.append(node)
    else:
        for child in node.nodes:
            _search_leafs(child, leafs)


def _search_best(nodes):
    # Nodes with the greatest reduction in the number of CNOTs.
    # There may be several with the same number.
    max_total_saved_cnots = max(nodes, key=lambda n: n.total_saved_cnots).total_saved_cnots
    max_saved_cnots_nodes = [node for node in nodes
                                if node.total_saved_cnots == max_total_saved_cnots]
    # Nodes with the minimum depth (wich depends on the size of the node's largest subsystem).
    # Shallower circuits with the same number of CNOTs means more parallelism.
    min_depth = _max_subsystem_size(min(max_saved_cnots_nodes, key=_max_subsystem_size))
    min_depth_nodes = [node for node in max_saved_cnots_nodes
                                if _max_subsystem_size(node) == min_depth]
    # Node with the lowest fidelity loss among the nodes with
    # the highest reduction in the number of CNOTs.
    return min(min_depth_nodes, key=lambda n: n.total_fidelity_loss)


def _max_subsystem_size(node):
    return len(max(node.qubits, key=len))


@numba.jit()
def _to_qubits(n_state_vector):
    return int(np.ceil(np.log2(n_state_vector))) if n_state_vector > 0 else 0


@numba.jit()
def idx_subsystem(idx: int, subsystem: np.ndarray):
    # The subsystem is a 1d array with numbers that represent the binary position in the binary representation
    # of idx. We need to build only the numbers from those binary positions.
    # We will give along the way an example
    # idx = 26 ==> 011010
    # subsystem = [1, 2, 4]
    # Thus we have X11X1X => 111 => 7
    subsystem_ordered: np.ndarray = subsystem.copy()
    subsystem_ordered.sort()
    subsystem_ordered = subsystem_ordered[::-1]

    bit_mask = 1 << subsystem_ordered       # => [010000, 001000, 000010] = [16, 8, 2]
    filtered = idx & bit_mask               #    [011010, 011010, 011010] => [010000, 001000, 000010] = [16, 8, 2]
    matches = filtered == bit_mask          #                             => [True,   True,  True  ]
    matches_int = matches.astype(np.float64)  #                             => [1,      1,     1     ]

    new_system = 2 ** np.arange(0, subsystem_ordered.shape[0], dtype=np.float64)[::-1]  # 2 ** [2, 1, 0] = [4, 2, 1]

    subsystem_idx = matches_int.dot(new_system)  # => [1,1, 1].[4, 2, 1] = 4 + 2 +  4 = 7
    return int(subsystem_idx)


# @numba.jit("complex128[:, :](complex128[:, :], int64[:])")
def _separation_matrix(vector: np.ndarray, subsystem2: np.ndarray):
    n_qubits = _to_qubits(vector.shape[0])
    subsystem1 = np.asarray(list(set(range(n_qubits)).difference(set(subsystem2))))

    new_shape = (2 ** subsystem1.shape[0], 2 ** subsystem2.shape[0])

    qubit_shape = tuple([2] * n_qubits)
    # We need to swap qubits from their subsystem2 position to the end of the
    # mode as we expect that we do LSB to be on the left -most side.
    from_move = subsystem2
    to_move = (n_qubits - np.arange(1, subsystem2.shape[0] + 1))[::-1]

    sep_matrix = np.moveaxis(vector.reshape(qubit_shape), from_move, to_move).reshape(new_shape)

    # sep_matrix = np.zeros(shape=new_shape, dtype=np.complex128)
    # for j, amp in enumerate(vector.flatten()):
    #     idx2_ = idx_subsystem(j, subsystem2)
    #     idx1_ = idx_subsystem(j, subsystem1)
    #     sep_matrix[idx1_, idx2_] = amp

    return sep_matrix


def _compute_schmidt(state_vector, entangled_qubits, qubits_to_disentangle, max_fidelity_loss, use_low_rank):
    # The following type casts and information are necessary because we use numba JIT compilation
    qubits_to_disentangle = np.asarray(qubits_to_disentangle, dtype=np.int64)
    entangled_qubits = np.asarray(entangled_qubits, dtype=np.int64)
    max_fidelity_loss = np.float64(max_fidelity_loss)
    state_vector = state_vector.reshape(-1, 1) if len(state_vector.shape) == 1 else state_vector
    node_fidelity_loss, subsystem1, subsystem2 = _compute_schmidt_jit(
        state_vector, entangled_qubits, qubits_to_disentangle, max_fidelity_loss, use_low_rank
    )
    node_fidelity_loss = node_fidelity_loss.flatten().real[0]
    subsystem1 = subsystem1.flatten() if subsystem1.shape[1] == 1 else subsystem1
    subsystem2 = subsystem2.flatten() if subsystem2.shape[1] == 1 else subsystem2
    return node_fidelity_loss, subsystem1, subsystem2


# @numba.jit('Tuple((complex128[:,:], complex128[:,:], complex128[:,:]))(complex128[:,:], int64[:], int64[:], float64, boolean)')
def _compute_schmidt_jit(state_vector, entangled_qubits: np.ndarray, qubits_to_disentangle, max_fidelity_loss, use_low_rank) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    # Maintains the relative position between the qubits of the two subsystems.
    # 1) the first statement is 'isin' which identifies those entries that are to disentangle
    # 2) where identifies the indices where where to disentangle
    # The return of where is a tuple, so we take the first element. The array must be of type int64!
    selected_qb = np.array([item in qubits_to_disentangle for item in entangled_qubits])
    local_qubits_to_disentangle = np.where(selected_qb)[0].astype(np.int64)

    # Create the matrix representation of the bi-partition
    sep_matrix = _separation_matrix(state_vector, local_qubits_to_disentangle)
    svd_u, svd_s, svd_v = np.linalg.svd(sep_matrix, full_matrices=False)

    # Find the best k-approx
    k = np.argmax(1 - np.cumsum(svd_s ** 2) < max_fidelity_loss)
    if k == 0 or k + 1 == svd_s.shape[0] or not use_low_rank:
        subsystem1_vector = svd_u[:, 0:1]
        subsystem2_vector = svd_v.T[:, 0:1]
        node_fidelity_loss = 1 - (svd_s ** 2)[0]  # svd_s first coefficient.
    else:
        node_fidelity_loss = 1 - sum((svd_s ** 2)[0:k+1])
        subsystem1_vector = svd_u[:, 0:k+1]
        subsystem2_vector = svd_v.T[:, 0:k+1]

    result = np.asarray([[node_fidelity_loss]], dtype=np.complex128), subsystem1_vector, subsystem2_vector
    return result


def _count_saved_cnots(entangled_vector, subsystem1_vector, subsystem2_vector):
    method = 'estimate'
    if len(subsystem1_vector.shape) > 1 and subsystem1_vector.shape[1] == subsystem2_vector.shape[1] > 1:
        cnots_new = schmidt_cnots(entangled_vector, low_rank=subsystem1_vector.shape[1], method=method)
    else:
        cnots_phase_3 = schmidt_cnots(subsystem1_vector, method=method)
        cnots_phase_4 = schmidt_cnots(subsystem2_vector, method=method)
        cnots_new = cnots_phase_3 + cnots_phase_4

    cnots_originally = schmidt_cnots(entangled_vector, method=method)

    return cnots_originally - cnots_new
