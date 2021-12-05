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
from itertools import combinations
from itertools import combinations, chain
from typing import List
import numpy as np

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
    n_qubits = int(np.log2(len(state_vector)))
    qubits = [list(range(n_qubits))]
    vectors = [state_vector]

    root_node = Node(0, 0, 0.0, 0.0, vectors, qubits, [0], [])
    _build_approximation_tree(root_node, max_fidelity_loss, strategy, max_combination_size, use_low_rank)

    leafs = []
    _search_leafs(root_node, leafs)

    best_node = _search_best(leafs)

    return best_node


def geometric_entanglement(state_vector):
    node = adaptive_approximation(state_vector, 1.0, strategy='greedy', max_combination_size=1, use_low_rank=False)
    return node.total_fidelity_loss


@dataclass
class Node:
    """
    Tree node used in _approximation_tree function
    """
    node_saved_cnots: int
    total_saved_cnots: int

    node_fidelity_loss: float
    total_fidelity_loss: float

    vectors: List[List[complex]]
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
            combs = _greedy_combinations(entangled_vector, entangled_qubits, max_k)
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
        if all([v.shape != entangled_vector.shape for v in new_node.vectors]):
            _build_approximation_tree(new_node, max_fidelity_loss, strategy, max_k)


def _all_combinations(entangled_qubits, max_k):
    return chain.from_iterable(combinations(entangled_qubits, k)
                                            for k in range(1, max_k+1))


def _greedy_combinations(entangled_vector, entangled_qubits, max_k):
    """
    Combinations with a qubit-by-qubit analysis.
    Returns only one representative of the bipartitions of size k (1<=k<=max_k).
    The increment in the partition size is done by choosing the qubit that has
    the lowest fidelity-loss when removed from the remaining entangled subsystem.
    """
    node = Node( 0, 0, 0.0, 0.0, [entangled_vector], [entangled_qubits], [] )
    for _ in range(max_k):
        current_vector = node.vectors[-1] # Last item is the current entangled state.
        current_qubits = node.qubits[-1]

        nodes = []
        # Disentangles one qubit at a time.
        for qubit_to_disentangle in current_qubits:
            node_fidelity_loss, subsystem1, subsystem2 = \
                _compute_schmidt(current_vector, current_qubits, [qubit_to_disentangle])

            new_node = _create_node(node, -1, [qubit_to_disentangle],
                                                node_fidelity_loss, subsystem1, subsystem2)
            nodes.append(new_node)
        # Search for the node with lowest fidelity-loss.
        node = _search_best(nodes)

    # Build the partitions by incrementing the number of selected qubits.
    # Returns only one partition for each length k.
    # All disentangled qubits are in the slice "node.qubits[0:max_k]", in the order in which
    # they were selected. Each partition needs to be ordered to ensure that the correct
    # construction of the circuit.
    return tuple( sorted( chain(*node.qubits[:k]) ) for k in range(1, max_k+1) )


def _compute_schmidt(state_vector, entangled_qubits, qubits_to_disentangle, max_fidelity_loss, use_low_rank):
    local_qubits_to_disentangle = []
    # Maintains the relative position between the qubits of the two subsystems.
    for qubit_to_disentangle in qubits_to_disentangle:
        local_qubits_to_disentangle.append(sum(i < qubit_to_disentangle for i in entangled_qubits))

    sep_matrix = _separation_matrix(state_vector, local_qubits_to_disentangle)
    svd_u, svd_s, svd_v = np.linalg.svd(sep_matrix, full_matrices=False)

    # Find the best k-approx
    k = np.argmax(1 - np.cumsum(svd_s ** 2) < max_fidelity_loss)
    if k == 0 or k + 1 == svd_s.shape[0] or not use_low_rank:
        subsystem1_vector = svd_u[:, 0]
        subsystem2_vector = svd_v.T[:, 0]
        node_fidelity_loss = 1 - (svd_s ** 2)[0]  # svd_s first coefficient.
    else:
        node_fidelity_loss = np.round(1 - sum((svd_s ** 2)[0:k+1]), decimals=13)
        subsystem1_vector = svd_u[:, 0:k+1]
        subsystem2_vector = svd_v.T[:, 0:k+1]

    return node_fidelity_loss, subsystem1_vector, subsystem2_vector


def _create_node(node, index, qubits_to_disentangle, node_fidelity_loss, subsystem1_vector, subsystem2_vector):
    total_fidelity_loss = 1 - (1 - node_fidelity_loss) * (1 - node.total_fidelity_loss)

    vectors = node.vectors.copy()
    qubits = node.qubits.copy()
    k_approximation  = node.k_approximation.copy()

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
    # Node with the lowest fidelity loss among the nodes with
    # the highest reduction in the number of CNOTs.
    return min(max_saved_cnots_nodes, key=lambda n: n.total_fidelity_loss)

def _separation_matrix(vector, subsystem2):
    n_qubits =  int(np.ceil(np.log2(vector.shape[0])))
    subsystem1 = list(set(range(n_qubits)).difference(set(subsystem2)))

    new_shape = (2 ** len(subsystem1), 2 ** len(subsystem2))

    sep_matrix = np.zeros(shape=new_shape, dtype=complex)

    for j, amp in enumerate(vector):
        current = f'{j:b}'.zfill(n_qubits)
        idx2 = ''.join([c for i, c in enumerate(current) if i in subsystem2])
        idx1 = ''.join([c for i, c in enumerate(current) if i in subsystem1])
        sep_matrix[int(idx1, 2), int(idx2, 2)] = amp

    return sep_matrix

def _count_saved_cnots(state_vector, entangled_vector, disentangled_vector):
    entangled_vector_rank = entangled_vector.shape[1] if len(entangled_vector.shape) == 2 else 0
    disentangled_vector_rank = disentangled_vector.shape[1] if len(disentangled_vector.shape) == 2 else 0

    cnots_phase_3 = _cnots_(_to_qubits(entangled_vector.shape[0]), entangled_vector_rank)
    cnots_phase_4 = _cnots_(_to_qubits(disentangled_vector.shape[0]), disentangled_vector_rank)
    cnots_originally = _cnots_(_to_qubits(state_vector.shape[0]))

    return cnots_originally - cnots_phase_3 - cnots_phase_4

def _cnots(n_qubits):
    if n_qubits == 1:
        return 0
    if n_qubits == 2:
        return 2
    if n_qubits == 3:
        return 4
    if n_qubits % 2 == 0:
        k = n_qubits/2
        return int(2 ** k - k - 1 + k + 23/24*2**(2*k) - 3/2 * 2**(k+1) + 8/3)

    k = (n_qubits-1)/2
    return int(2 ** k - k - 1 + k + 23/48*2**(2*k) - 3/2 * 2**(k) + 4/3 +
                                    23/48*2**(2*k + 2) - 3/2 * 2**(k + 1) + 4/3)

def _cnots_(n_qubits, rank=0):
    if n_qubits % 2 == 0:
        k = n_qubits // 2
        m = _to_qubits(rank) if rank > 1 else k
        if m < k:
        # Isometries
            phase_1 = 2 ** m - m - 1
            phase_2 = m
            phase_3 = 2**(m+k) - 1/24 * 2**k
            phase_4 = 2**(m+k) - 1/24 * 2**k
        else:
            # Unitaries
            phase_1 = 2 ** m - m - 1
            phase_2 = m
            phase_3 = 23 / 48 * 2 ** (2 * k) - 3 / 2 * 2 ** k + 4 / 3
            phase_4 = 23 / 48 * 2 ** (2 * k) - 3 / 2 * 2 ** k + 4 / 3
    else:
        k = (n_qubits - 1) // 2
        m = _to_qubits(rank) if rank > 1 else k
        if m < k:
            # Isometries
            phase_1 = 2 ** m - m - 1
            phase_2 = m
            phase_3 = 2**(m+k) - 1/24 * 2**k
            phase_4 = 2**(m+k+1) - 1/24 * 2**(k+1)
        else:
            # Unitaries
            phase_1 = 2 ** m - m - 1
            phase_2 = m
            phase_3 = 23 / 48 * 2 ** (2 * k) - 3 / 2 * 2 ** k + 4 / 3
            phase_4 = 23 / 48 * 2 ** (2 * (k + 1)) - 3 / 2 * 2 ** (k + 1) + 4 / 3

    return int(np.floor(phase_1 + phase_2 + phase_3 + phase_4))


def _to_qubits(n_state_vector):
    return int(np.ceil(np.log2(n_state_vector)))
