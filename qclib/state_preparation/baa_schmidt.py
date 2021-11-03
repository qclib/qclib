import itertools
import logging
from typing import List, Tuple, Optional, Union

import numpy as np
import qiskit
from numpy import ndarray

from qclib.state_preparation import schmidt

LOG = logging.getLogger(__name__)


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
    vectors: List[ndarray]
    cnot_saving: int
    fidelity_loss: float

    def __init__(self, split_program: Tuple[Split, ...], vectors: List[np.ndarray], fidelity_loss: float, cnot_saving: int):
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


def sp_cnots(n) -> int:
    if n == 1:
        return 0
    elif n == 2:
        return 2
    elif n == 3:
        return 4
    elif n % 2 == 0:
        k = n/2
        return int(2 ** k - k - 1 + k + 23/24*2**(2*k) - 3/2 * 2**(k+1) + 8/3)
    else:
        k = (n-1)/2
        return int(2 ** k - k - 1 + k + 23/48*2**(2*k) - 3/2 * 2**(k) + 4/3 + 23/48*2**(2*k + 2) - 3/2 * 2**(k + 1) + 4/3)


def to_qubits(d):
    return int(np.ceil(np.log2(d)))


def get_complementary_subsystem(subsystem: Tuple[int, ...], num_qubits: int):
    subsystem_c = tuple(set(range(num_qubits)).difference(set(subsystem)))
    return subsystem_c


def get_separation_matrix(vector: np.ndarray, subsystem_2: Tuple[int, ...]):
    num_qubits =  int(np.ceil(np.log2(vector.shape[0])))
    subsystem_1 = get_complementary_subsystem(subsystem_2, num_qubits)

    new_shape = (2 ** len(subsystem_1), 2 ** len(subsystem_2))
    M = np.zeros(shape=new_shape, dtype=complex)

    for n, v in enumerate(vector):
        current = f'{n:b}'.zfill(num_qubits)
        number_2 = ''.join([c for i, c in enumerate(current) if i in subsystem_2])
        number_1 = ''.join([c for i, c in enumerate(current) if i in subsystem_1])
        M[int(number_1, 2), int(number_2, 2)] = v

    return M


def get_bipartite_systems(vector: np.ndarray) -> List[Tuple[int, ...]]:
    # Bundle the state vector into a qutip Qobj
    num_qb = int(np.ceil(np.log2(vector.shape[0])))

    size_biggest_coef = []
    # This loop looks at the principal subsystem size: starting from 1 to at most the half of the system
    # (+1 to have in included). The reason is that HS decomposition says that the EVs are the same for the 'other'
    # system.
    for size in range(1, num_qb // 2 + 1):
        sub_systems = itertools.combinations(range(num_qb), size)
        size_biggest_coef += sub_systems
    return size_biggest_coef


def get_nodes_from_activations(vectors: List[np.ndarray], subsystem_list: List[List[Tuple[int, ...]]],
                               activations: np.ndarray) -> List[Node]:
    result = []
    # This step is the main step, I need to explain this a bit better!
    all_paths = [[Split(dd, -1.0) for dd in d] if a == 1 else [Split(None, 0.0)] for d, a in
                 zip(subsystem_list, activations)]

    # The product makes all possible cross-product combinations as given above. Need to explain this better!
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
                # Compute the Schmidt States
                M = get_separation_matrix(vector, split.subsystem)
                U, S, Vh = np.linalg.svd(M, full_matrices=False)
                vector_1 = U[:, 0]
                vector_2 = Vh.T[:, 0]
                new_vectors.append(vector_1)
                new_vectors.append(vector_2)
                cnots_phase_3 = sp_cnots(to_qubits(vector_1.shape[0]))
                cnots_phase_4 = sp_cnots(to_qubits(vector_2.shape[0]))
                cnots_originally = sp_cnots(to_qubits(vector.shape[0]))
                saved_cnots += cnots_originally - cnots_phase_3 - cnots_phase_4
                split.fidelity_loss = 1 - (S ** 2)[0]
                new_fidelity *= 1 - split.fidelity_loss
        result.append(Node(split_program, new_vectors, 1 - np.around(new_fidelity, 6), saved_cnots))
    return result


def generate_subsystem_partitions(vectors: List[np.ndarray], no_branching=False):
    subsystem_list: List[List[Tuple[int], ...]] = [get_bipartite_systems(vector) for vector in vectors]

    result = []
    v = np.zeros(len(vectors))
    v[0] = 1
    if no_branching:
        result += get_nodes_from_activations(vectors, [[d[0]] for d in subsystem_list if len(d) > 0], v)
    else:
        for activations in [np.roll(v, i) for i in range(len(vectors))]:
            if set(activations) == {0}:
                continue
            result += get_nodes_from_activations(vectors, subsystem_list, activations)

    return result


def search_best_node(vectors: List[ndarray], running_cnot_saving: int, running_fidelity_loss: float, max_fidelity_loss: float) -> Optional[Node]:
    data: List[Node] = generate_subsystem_partitions(vectors)
    possible_data: List[Node] = [d for d in data if 1 - (1 - d.fidelity_loss) * (1 - running_fidelity_loss) <= max_fidelity_loss]
    better = [search_best_node(p.vectors, running_cnot_saving + p.cnot_saving, running_fidelity_loss + p.fidelity_loss, max_fidelity_loss) for p in possible_data]
    if len(better) == 0 or (set(better) == {None}):
        if len(possible_data) == 0:
            return None
        else:
            best_possible = max(possible_data, key=lambda n: n.cnot_saving)
            total_fidelity_loss = 1 - (1 - best_possible.fidelity_loss) * (1 - running_fidelity_loss)
            total_cnot_savings = running_cnot_saving + best_possible.cnot_saving
            return Node((), best_possible.vectors, total_fidelity_loss, total_cnot_savings)
    else:
        return max([b for b in better if b is not None], key=lambda n: n.cnot_saving)


def adaptive_approximation(vector: np.ndarray, max_fidelity_loss: float) -> Optional[Node]:
    best_node: Optional[Node] = search_best_node([vector], 0, 0.0, max_fidelity_loss)
    LOG.debug(f'Best Node: {best_node}')
    return best_node


def initialize(state_vector, max_fidelity_loss=0.0, isometry_scheme='ccd', unitary_scheme='qsd'):
    """ State preparation using the bounded approximation algorithm via Schmidt decomposition arXiv:1003.5760

        For instance, to initialize the state a|0> + b|1>
            $ state = [a, b]
            $ circuit = initialize(state)

        Parameters
        ----------
        state_vector: list of float or array-like
            A unit vector representing a quantum state.
            Values are amplitudes.

        max_fidelity_loss: float
            ``state`` allowed (fidelity) error for approximation (0 <= ``max_fidelity_loss`` <= 1).
            If ``max_fidelity_loss`` is not in the valid range, it will be ignored.

        isometry_scheme: string
            Scheme used to decompose isometries.
            Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
            Default is ``isometry_scheme='ccd'``.

        unitary_scheme: string
            Scheme used to decompose unitaries.
            Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
            Shannon decomposition).
            Default is ``unitary_scheme='qsd'``.

        Returns
        -------
        circuit: QuantumCircuit
            QuantumCircuit to initialize the state.
        """
    if max_fidelity_loss > 1 or max_fidelity_loss < 0:
        max_fidelity_loss = 0.0

    node_option: Optional[Node] = adaptive_approximation(state_vector, max_fidelity_loss)
    if node_option is None:
        return schmidt.initialize(state_vector, low_rank=-1, isometry_scheme=isometry_scheme, unitary_scheme=unitary_scheme)

    num_qubits = to_qubits(len(state_vector))
    qc = qiskit.QuantumCircuit(num_qubits)
    offset = 0
    for vec in node_option.vectors:
        vec_num_qubits = to_qubits(len(vec))
        if vec_num_qubits == 1:
            qc_vec = qiskit.QuantumCircuit(1)
            qc_vec.initialize(vec)
        else:
            qc_vec = schmidt.initialize(vec, low_rank=-1, isometry_scheme=isometry_scheme, unitary_scheme=unitary_scheme)
        affected_qubits = list(range(offset, offset + vec_num_qubits))
        qc = qc.compose(qc_vec, affected_qubits)
        offset += vec_num_qubits
    return qc
