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

from typing import List, Tuple, Optional
import itertools
import numpy as np

# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring

class Split:
    fidelity_loss: float
    subsystem: Optional[Tuple[int, ...]]

    def __init__(self, subsystem: Optional[Tuple[int, ...]], fidelity_loss: float):
        self.subsystem = subsystem
        self.fidelity_loss = fidelity_loss

    def __str__(self):
        return f'{type(self).__name__}|{self.subsystem}|'

    def __repr__(self):
        return str(self)

class Node:
    split_program: Tuple[Split, ...]
    vectors: List[np.ndarray]
    cnot_saving: int
    fidelity_loss: float

    def __init__(self, split_program: Tuple[Split, ...], vectors: List[np.ndarray],
                    fidelity_loss: float, cnot_saving: int):
        self.fidelity_loss = fidelity_loss
        self.cnot_saving = cnot_saving
        self.vectors = vectors
        self.split_program = split_program

    def __getitem__(self, item):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return data[item]

    def __iter__(self):
        data = [self.split_program, self.vectors, self.fidelity_loss, self.cnot_saving]
        return iter(data)

    def __str__(self):
        return f'Node{(self.split_program, self.fidelity_loss, self.cnot_saving, self.vectors)}'

    def __repr__(self):
        return str(self)

def adaptive_approximation(vector: np.ndarray, max_fidelity_loss: float) -> Optional[Node]:
    best_node: Optional[Node] = _search_best_node([vector], 0, 0.0, max_fidelity_loss)
    return best_node




def _sp_cnots(n_qubits) -> int:
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

def _to_qubits(n_state_vector):
    return int(np.ceil(np.log2(n_state_vector)))

def _get_complementary_subsystem(subsystem: Tuple[int, ...], num_qubits: int):
    subsystem_c = tuple(set(range(num_qubits)).difference(set(subsystem)))
    return subsystem_c

def _get_separation_matrix(vector: np.ndarray, subsystem_2: Tuple[int, ...]):
    n_qubits =  int(np.ceil(np.log2(vector.shape[0])))
    subsystem_1 = _get_complementary_subsystem(subsystem_2, n_qubits)

    new_shape = (2 ** len(subsystem_1), 2 ** len(subsystem_2))
    sep_matrix = np.zeros(shape=new_shape, dtype=complex)

    for i, amp in enumerate(vector):
        current = f'{i:b}'.zfill(n_qubits)
        number_2 = ''.join([c for i, c in enumerate(current) if i in subsystem_2])
        number_1 = ''.join([c for i, c in enumerate(current) if i in subsystem_1])
        sep_matrix[int(number_1, 2), int(number_2, 2)] = amp

    return sep_matrix

def _get_bipartite_systems(vector: np.ndarray) -> List[Tuple[int, ...]]:
    # Bundle the state vector into a qutip Qobj
    num_qb = int(np.ceil(np.log2(vector.shape[0])))

    size_biggest_coef = []
    # This loop looks at the principal subsystem size: starting from 1 to at most
    # the half of the system (+1 to have in included).
    # The reason is that HS decomposition says that the EVs are the same for the 'other' system.
    for size in range(1, num_qb // 2 + 1):
        sub_systems = itertools.combinations(range(num_qb), size)
        size_biggest_coef += sub_systems
    return size_biggest_coef

def _compute_schmidt_states(vector, split, new_vectors):
    sep_matrix = _get_separation_matrix(vector, split.subsystem)
    svd_u, svd_s, svd_v = np.linalg.svd(sep_matrix, full_matrices=False)
    vector_1 = svd_u[:, 0]
    vector_2 = svd_v.T[:, 0]
    new_vectors.append(vector_1)
    new_vectors.append(vector_2)
    cnots_phase_3 = _sp_cnots(_to_qubits(vector_1.shape[0]))
    cnots_phase_4 = _sp_cnots(_to_qubits(vector_2.shape[0]))
    cnots_originally = _sp_cnots(_to_qubits(vector.shape[0]))
    saved_cnots = cnots_originally - cnots_phase_3 - cnots_phase_4
    split.fidelity_loss = 1 - (svd_s ** 2)[0]

    return saved_cnots

def _get_nodes_from_activations(vectors: List[np.ndarray],
                                subsystem_list: List[List[Tuple[int, ...]]],
                                activations: np.ndarray) -> List[Node]:
    result = []
    # This step is the main step, I need to explain this a bit better!
    all_paths = [[Split(dd, -1.0) for dd in d] if a == 1 else [Split(None, 0.0)] for d, a in
                 zip(subsystem_list, activations)]

    # The product makes all possible cross-product combinations as given above.
    # Need to explain this better!
    for split_program in itertools.product(*all_paths):
        # apply the split program to the vectors and generate the new children
        new_vectors = []
        new_fidelity = 1.0
        saved_cnots = 0
        split: Split
        for vector, split in zip(vectors, split_program):
            if split.subsystem is None:
                new_vectors.append(vector)
            else:
                saved_cnots += _compute_schmidt_states(vector, split, new_vectors)
                new_fidelity *= 1 - split.fidelity_loss
        result.append(Node(split_program, new_vectors, 1 - np.around(new_fidelity, 6),
                                                                            saved_cnots))
    return result

def _generate_subsystem_partitions(vectors: List[np.ndarray], no_branching=False):
    subsystem_list: List[List[Tuple[int], ...]] = \
        [_get_bipartite_systems(vector) for vector in vectors]

    result = []
    activations = np.zeros(len(vectors))
    activations[0] = 1
    if no_branching:
        result += _get_nodes_from_activations(vectors,
                    [[d[0]] for d in subsystem_list if len(d) > 0], activations)
    else:
        for activations in [np.roll(activations, i) for i in range(len(vectors))]:
            if set(activations) == {0}:
                continue
            result += _get_nodes_from_activations(vectors, subsystem_list, activations)

    return result

def _search_best_node(vectors: List[np.ndarray], running_cnot_saving: int,
                        running_fidelity_loss: float, max_fidelity_loss: float) -> Optional[Node]:
    data: List[Node] = _generate_subsystem_partitions(vectors)
    possible_data: List[Node] = [d for d in data
                                    if 1 - (1 - d.fidelity_loss) * (1 - running_fidelity_loss)
                                        <= max_fidelity_loss]
    better = [_search_best_node(p.vectors, running_cnot_saving + p.cnot_saving,
                                running_fidelity_loss + p.fidelity_loss, max_fidelity_loss)
                                for p in possible_data]
    if len(better) == 0 or (set(better) == {None}):
        if len(possible_data) == 0:
            return None

        best_possible = max(possible_data, key=lambda n: n.cnot_saving)
        total_fidelity_loss = 1 - (1 - best_possible.fidelity_loss) * \
                                    (1 - running_fidelity_loss)
        total_cnot_savings = running_cnot_saving + best_possible.cnot_saving
        return Node((), best_possible.vectors, total_fidelity_loss, total_cnot_savings)

    return max([b for b in better if b is not None], key=lambda n: n.cnot_saving)
