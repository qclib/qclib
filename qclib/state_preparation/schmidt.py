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
Implements the state preparation
defined at https://arxiv.org/abs/1003.5760.
"""

from math import ceil, log2
import numpy as np
import deprecation
from qiskit import QuantumCircuit, QuantumRegister
from qclib.state_preparation.mottonen import initialize as mottonen
from qclib.unitary import unitary as decompose_unitary, cnot_count as cnots_unitary
from qclib.isometry import decompose as decompose_isometry, cnot_count as cnots_isometry

# pylint: disable=maybe-no-member


def initialize(state_vector, low_rank=0, isometry_scheme='ccd',
               unitary_scheme='qsd', partition=None):
    """
    Low-rank state preparation using Schmidt decomposition.
    https://arxiv.org/abs/2111.03132

    For instance, to initialize the state a|00> + b|10> (|a|^2+|b|^2=1)
        $ state = [a, 0, b, 0]
        $ circuit = initialize(state)

    Parameters
    ----------
    state_vector: list of complex
        A unit vector representing a quantum state.
        Values are amplitudes.

    low_rank: int
        ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
        If ``low_rank`` is not in the valid range, it will be ignored.
        This parameter limits the rank of the Schmidt decomposition. If the Schmidt rank
        of the state decomposition is greater than ``low_rank``, a low-rank approximation
        is applied.

    isometry_scheme: string
        Scheme used to decompose isometries.
        Possible values are ``'knill'`` and ``'ccd'`` (column-by-column decomposition).
        Default is ``isometry_scheme='ccd'``.

    unitary_scheme: string
        Scheme used to decompose unitaries.
        Possible values are ``'csd'`` (cosine-sine decomposition) and ``'qsd'`` (quantum
        Shannon decomposition).
        Default is ``unitary_scheme='qsd'``.

    partition: list of int
        Set of qubit indices that represent a part of the bipartition.
        The other partition will be the relative complement of the full set of qubits
        with respect to the set ``partition``.
        The valid range for indexes is ``0 <= index < n_qubits``. The number of indexes
        in the partition must be greater than or equal to ``1`` and less than or equal
        to ``n_qubits//2`` (``n_qubits//2+1`` if ``n_qubits`` is odd).
        Default is ``partition=list(range(n_qubits//2 + odd))``.

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """

    n_qubits = _to_qubits(len(state_vector))
    if n_qubits < 2:
        return mottonen(state_vector)

    circuit, reg_a, reg_b = _create_quantum_circuit(state_vector, partition)

    # Schmidt decomposition
    svd_u, singular_values, svd_v = schmidt_decomposition(state_vector, reg_a)

    rank, svd_u, svd_v, singular_values = \
        low_rank_approximation(low_rank, svd_u, svd_v, singular_values)

    # Schmidt measure of entanglement
    ebits = _to_qubits(rank)

    # Phase 1. Encodes the singular values.
    if ebits > 0:
        reg_sv = reg_b[:ebits]
        singular_values = singular_values / np.linalg.norm(singular_values)
        _encode(singular_values.reshape(rank, 1), circuit, reg_sv,
                isometry_scheme, unitary_scheme)

    # Phase 2. Entangles only the necessary qubits, according to rank.
    for j in range(ebits):
        circuit.cx(reg_b[j], reg_a[j])

    # Phase 3 and 4 encode gates U and V.T
    _encode(svd_u, circuit, reg_b, isometry_scheme, unitary_scheme)
    _encode(svd_v.T, circuit, reg_a, isometry_scheme, unitary_scheme)

    return circuit.reverse_bits()


def schmidt_decomposition(state_vector, partition):
    """
    Execute the Schmidt decomposition of a state vector.

    Parameters
    ----------
    state_vector: list of complex
        A unit vector representing a quantum state.
        Values are amplitudes.

    partition: list of int
        Set of qubit indices that represent a part of the bipartition.
        The other partition will be the relative complement of the full set of qubits
        with respect to the set ``partition``.
        The valid range for indexes is ``0 <= index < n_qubits``. The number of indexes
        in the partition must be greater than or equal to ``1`` and less than or equal
        to ``n_qubits//2`` (``n_qubits//2+1`` if ``n_qubits`` is odd).
    """

    n_qubits = _to_qubits(len(state_vector))

    sep_matrix = _separation_matrix(n_qubits, state_vector, partition)

    return np.linalg.svd(sep_matrix)


def schmidt_composition(svd_u, svd_v, singular_values, partition):
    """
    Execute the Schmidt composition of a state vector.
    The inverse of the Schmidt decomposition.

    Returns
    -------
    state_vector: list of complex
        A unit vector representing a quantum state.
        Values are amplitudes.
    """

    n_qubits = _to_qubits(svd_u.shape[0]) + _to_qubits(svd_v.shape[1])

    sep_matrix = (svd_u * singular_values) @ svd_v

    state_vector = _undo_separation_matrix(n_qubits, sep_matrix, partition)

    return state_vector


def low_rank_approximation(low_rank, svd_u, svd_v, singular_values):
    """
    Low-rank approximation from the SVD.
    """
    rank = singular_values.shape[0]  # max rank

    effective_rank = _effective_rank(singular_values)

    if 0 < low_rank < rank or effective_rank < rank:
        if 0 < low_rank < effective_rank:
            effective_rank = low_rank

        # To use isometries, the rank needs to be a power of 2.
        rank = int(2**ceil(log2(effective_rank)))

        svd_u = svd_u[:, :rank]
        svd_v = svd_v[:rank, :]
        singular_values = singular_values[:rank]

    return rank, svd_u, svd_v, singular_values


def _separation_matrix(n_qubits, state_vector, partition):
    new_shape = (2 ** (n_qubits-len(partition)), 2 ** len(partition))

    qubit_shape = tuple([2] * n_qubits)
    # We need to swap qubits from their subsystem2 position to the end of the
    # mode as we expect that we do LSB to be on the left-most side.
    from_move = sorted(partition)
    to_move = (n_qubits - np.arange(1, len(partition) + 1))[::-1]

    sep_matrix = \
        np.moveaxis(np.array(state_vector).reshape(qubit_shape),
                    from_move, to_move).reshape(new_shape)
    return sep_matrix


def _undo_separation_matrix(n_qubits, sep_matrix, partition):
    new_shape = (2 ** n_qubits, )

    qubit_shape = tuple([2] * n_qubits)

    to_move = sorted(partition)
    from_move = (n_qubits - np.arange(1, len(partition) + 1))[::-1]

    state_vector = \
        np.moveaxis(np.array(sep_matrix).reshape(qubit_shape),
                    from_move, to_move).reshape(new_shape)
    return state_vector


def _effective_rank(singular_values):
    return sum(j > 10**-7 for j in singular_values)


def _to_qubits(n_state_vector):
    return int(log2(n_state_vector))


def _default_partition(n_qubits):
    odd = n_qubits % 2
    return list(range(n_qubits//2 + odd))


def _create_quantum_circuit(state_vector, partition):
    n_qubits = _to_qubits(len(state_vector))
    if partition is None:
        partition = _default_partition(n_qubits)

    complement = sorted(set(range(n_qubits)).difference(set(partition)))

    circuit = QuantumCircuit(n_qubits)

    return circuit, partition[::-1], complement[::-1]


def _encode(data, circuit, reg, iso_scheme='ccd', uni_scheme='qsd'):
    """
    Encodes data using the most appropriate method.
    """
    n_qubits = len(reg)
    rank = 0
    if data.shape[1] == 1:
        partition = _default_partition(n_qubits)
        _, svals, _ = schmidt_decomposition(data[:, 0], partition)
        rank = _effective_rank(svals)

    if data.shape[1] == 1 and (n_qubits % 2 == 0 or n_qubits < 4 or rank == 1):
        # state preparation
        gate_u = initialize(data[:, 0], isometry_scheme=iso_scheme,
                            unitary_scheme=uni_scheme)

    elif data.shape[0] > data.shape[1]:
        gate_u = decompose_isometry(data, scheme=iso_scheme)
    else:
        gate_u = decompose_unitary(data, decomposition=uni_scheme)

    # Apply gate U to the register reg
    circuit.compose(gate_u, reg, inplace=True)


def cnot_count(state_vector, low_rank=0, isometry_scheme='ccd', unitary_scheme='qsd',
               partition=None, method='estimate'):
    """
    Estimate the number of CNOTs to build the state preparation circuit.
    """

    n_qubits = _to_qubits(len(state_vector))
    if n_qubits < 2:
        return 0

    if partition is None:
        partition = _default_partition(n_qubits)

    cnots = 0

    # Schmidt decomposition
    svd_u, singular_values, svd_v = schmidt_decomposition(state_vector, partition)

    rank, svd_u, svd_v, singular_values = \
        low_rank_approximation(low_rank, svd_u, svd_v, singular_values)

    # Schmidt measure of entanglement
    ebits = _to_qubits(rank)

    # Phase 1.
    if ebits > 0:
        singular_values = singular_values / np.linalg.norm(singular_values)
        cnots += _cnots(singular_values.reshape(rank, 1), isometry_scheme,
                        unitary_scheme, method)
    # Phase 2.
    cnots += ebits

    # Phases 3 and 4.
    cnots += _cnots(svd_u, isometry_scheme, unitary_scheme, method)
    cnots += _cnots(svd_v.T, isometry_scheme, unitary_scheme, method)

    return cnots


def _cnots(data, iso_scheme='ccd', uni_scheme='qsd', method='estimate'):
    n_qubits = _to_qubits(data.shape[0])

    rank = 0
    if data.shape[1] == 1:
        partition = _default_partition(n_qubits)
        _, svals, _ = schmidt_decomposition(data[:, 0], partition)
        rank = _effective_rank(svals)

    if data.shape[1] == 1 and (n_qubits % 2 == 0 or n_qubits < 4 or rank == 1):
        return cnot_count(data[:, 0], isometry_scheme=iso_scheme,
                          unitary_scheme=uni_scheme, method=method)

    if data.shape[0] > data.shape[1]:
        return cnots_isometry(data, scheme=iso_scheme, method=method)

    return cnots_unitary(data, decomposition=uni_scheme, method=method)


@deprecation.deprecated(deprecated_in="0.0.7",
                        details="Use the initialize function instead")
def initialize_original(state_vector):
    """ State preparation using Schmidt decomposition arXiv:1003.5760.
        This function implements the original algorithm as defined in arXiv:1003.5760.
        It is kept here for didactic reasons.
        The ``initialize`` function should preferably be used.

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize_original(state)

    Parameters
    ----------
    state_vector: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """

    state = np.copy(state_vector)

    size = len(state)
    n_qubits = _to_qubits(size)

    odd = n_qubits % 2

    state.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + odd)))

    svd_u, singular_values, svd_v = np.linalg.svd(state)

    singular_values = singular_values / np.linalg.norm(singular_values)

    reg_a = QuantumRegister(n_qubits//2 + odd)
    reg_b = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(reg_a, reg_b)

    if len(singular_values) > 2:
        circ = initialize_original(singular_values)
    else:
        circ = mottonen(singular_values)
    circuit.compose(circ, reg_b, inplace=True)

    for k in range(int(n_qubits//2)):
        circuit.cx(reg_b[k], reg_a[k])

    gate_u = decompose_unitary(svd_u, 'qsd')
    gate_v = decompose_unitary(svd_v.T, 'qsd')

    circuit.compose(gate_u, reg_b, inplace=True)  # apply gate U to the first register
    circuit.compose(gate_v, reg_a, inplace=True)  # apply gate V to the second register

    return circuit
