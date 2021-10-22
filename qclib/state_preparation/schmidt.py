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

import numpy as np
import deprecation
from qiskit import QuantumCircuit, QuantumRegister
from qclib.state_preparation.mottonen import initialize as mottonen
from qclib.unitary import unitary as decompose_unitary
from qclib.isometry import decompose as decompose_isometry

# pylint: disable=maybe-no-member


def initialize(state_vector, low_rank=0, isometry_scheme='ccd', unitary_scheme='qsd'):
    """ State preparation using Schmidt decomposition arXiv:1003.5760

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize(state)

    Parameters
    ----------
    state_vector: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    low_rank: int
        ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
        If ``low_rank`` is not in the valid range, it will be ignored.

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

    n_qubits, odd, rank, singular_values, svd_u, svd_v = _rank_aprox(low_rank, state_vector)

    reg_a = QuantumRegister(n_qubits//2 + odd)
    reg_b = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(reg_a, reg_b)

    if len(singular_values) > 2:
        circ = initialize(singular_values)
    else:
        circ = mottonen(singular_values)

    circuit.compose(circ, reg_b[:int(np.log2(len(singular_values)))], inplace=True)

    for j in range(int(np.log2(rank))):                # Entangles only the necessary qubits,
        circuit.cx(reg_b[j], reg_a[j])                     # according to rank.

    _encode(svd_u, circuit, reg_b, isometry_scheme, unitary_scheme)
    _encode(svd_v.T, circuit, reg_a, isometry_scheme, unitary_scheme)

    return circuit


def _rank_aprox(low_rank, state_vector):
    state = np.copy(state_vector)
    size = len(state)
    n_qubits = int(np.log2(size))
    odd = n_qubits % 2
    lines = int(2 ** (n_qubits // 2))
    cols = int(2 ** (n_qubits // 2 + odd))
    state.shape = (lines, cols)
    svd_u, singular_values, svd_v = np.linalg.svd(state)
    rank = lines
    if 0 < low_rank < rank:
        e_rank = sum(j > 10 ** -16 for j in singular_values)  # Effective rank.
        if low_rank < e_rank:
            e_rank = low_rank  # Low-rank approximation

        rank = int(2 ** np.ceil(np.log2(e_rank)))  # To use isometries, the rank needs to be
        # a power of 2.

        svd_u = svd_u[:, :rank]  # Matrix u can be a unitary (k=n) or
        # isometry (k<n).

        svd_v = svd_v[:rank, :]  # If n<m, v.T is always an isometry of
        # log2(k) to log2(m) (rank<=n).
        # If n=m, v.T can be a unitary (k=n) or
        # isometry (k<n).

        singular_values = singular_values[:rank]  # The length of the state vector needs to
        # be a power of 2.

        singular_values[e_rank:] = np.zeros(rank - e_rank)  # If k>rank, zeroes out
        # the additional elements.
        if e_rank == 1:
            singular_values = np.concatenate((singular_values, [0]))  # The length of the state
            # vector needs to be a
            # power of 2 and >1.
    singular_values = singular_values / np.linalg.norm(singular_values)
    return n_qubits, odd, rank, singular_values, svd_u, svd_v


def _encode(unitary, circuit, reg, isometry_scheme, unitary_scheme):
    """ Encodes the data using the most appropriate state preparation method """

    if unitary.shape[1] == 1:
        if unitary.shape[0] > 2:
            gate_u = initialize(unitary[:, 0])
        else:
            gate_u = mottonen(unitary[:, 0])
    elif unitary.shape[0] > unitary.shape[1]:  # Isometry decomposition.
        gate_u = decompose_isometry(unitary, scheme=isometry_scheme)
    else:  # Unitary decomposition.
        gate_u = decompose_unitary(unitary, decomposition=unitary_scheme)

    circuit.compose(gate_u, reg, inplace=True)  # Apply gate U to the register.


def _change_toschmidt_basis(circuit, reg_a, reg_b, svd_u, svd_v):
    gate_u = decompose_unitary(svd_u, 'qsd')
    gate_v = decompose_unitary(svd_v.T, 'qsd')
    circuit.compose(gate_u, reg_b, inplace=True)  # apply gate U to the first register
    circuit.compose(gate_v, reg_a, inplace=True)  # apply gate V to the second register


def _initialize_singular_values(circuit, reg_b, singular_values):
    if len(singular_values) > 2:
        circ = initialize_original(singular_values)
    else:
        circ = mottonen(singular_values)
    circuit.compose(circ, reg_b, inplace=True)


@deprecation.deprecated(deprecated_in="0.0.7",
                        details="Use the initialize function instead")
def initialize_original(state_vector):
    """ State preparation using Schmidt decomposition arXiv:1003.5760.
        This function implements the original algorithm as defined in arXiv:1003.5760.
        It is kept here for didactic reasons.
        The ``initialize`` function should preferably be used.
        This function is used in isometry.py.

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
    n_qubits = int(np.log2(size))

    odd = n_qubits % 2

    state.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + odd)))

    svd_u, singular_values, svd_v = np.linalg.svd(state)

    singular_values = singular_values / np.linalg.norm(singular_values)

    reg_a = QuantumRegister(n_qubits//2 + odd)
    reg_b = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(reg_a, reg_b)

    _initialize_singular_values(circuit, reg_b, singular_values)

    for k in range(int(n_qubits//2)):
        circuit.cx(reg_b[k], reg_a[k])

    _change_toschmidt_basis(circuit, reg_a, reg_b, svd_u, svd_v)

    return circuit
