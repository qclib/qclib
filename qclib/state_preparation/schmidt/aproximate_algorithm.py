import datetime
import itertools
import logging
from typing import List, Tuple, Optional

import numpy as np
import qiskit
from numpy import ndarray
from qiskit.circuit.random import random_circuit
from qiskit.providers import aer

from qclib.state_preparation.schmidt.models import Node, Split
from qclib.state_preparation.util.entanglement_measure import calculate_entropy_meyer_wallach, compute_Q_ptrace

logging.basicConfig(format='%(asctime)s::' + logging.BASIC_FORMAT, level='ERROR')
LOG = logging.getLogger(__name__)


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
                # k = to_qubits(min(vector_1.shape[0], vector_2.shape[0]))
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
    LOG.debug('Creating Graph')
    # fidelity_loss, product_state = get_fidelity_loss(vector, return_product_state=True)

    # if fidelity_loss <= max_fidelity_loss:
    #     LOG.debug(f'Product State is within limits {fidelity_loss} <= {max_fidelity_loss}.')
    #     return product_state

    best_node: Optional[Node] = search_best_node([vector], 0, 0.0, max_fidelity_loss)
    LOG.debug(f'Best Node: {best_node}')
    return best_node


if __name__ == "__main__":
    exp_time_start = datetime.datetime.now()
    LOG.setLevel('INFO')
    num_qubits = 4
    mw_limit_lower = 0.5
    mw_limit_upper = 0.6
    for _ in range(10):
        mw = -1.0
        while mw < mw_limit_lower or mw > mw_limit_upper:
            qc: qiskit.QuantumCircuit = random_circuit(num_qubits, 2*num_qubits)
            job: aer.AerJob = qiskit.execute(qc, backend=aer.StatevectorSimulator())
            vector = job.result().get_statevector()
            mw = compute_Q_ptrace(vector)
            assert abs(mw - calculate_entropy_meyer_wallach(vector)) < 1e-3

        LOG.debug(f"The Circuit\n{qc.draw(fold=-1)}")
        LOG.debug(f"Vector: {np.linalg.norm(vector)}\n {vector}")
        LOG.debug(f"Meyer-Wallach: {mw}.")

        start = datetime.datetime.now()
        max_fidelity_loss = 0.1
        node = adaptive_approximation(vector, max_fidelity_loss)
        end = datetime.datetime.now()

        if node is None:
            LOG.info(f'[{max_fidelity_loss}] No approximation could be found (MW: {mw}). ({end - start})')
        else:
            moettoenen_sp = sum([2**n for n in range(1, num_qubits)])
            LOG.info(f'[{max_fidelity_loss}] With fidelity loss {node.fidelity_loss} (MW: {mw}) we can '
                     f'save {node.cnot_saving} of {sp_cnots(num_qubits)} (Moettonen:{moettoenen_sp}) CNOT-gates. ({end - start})')
