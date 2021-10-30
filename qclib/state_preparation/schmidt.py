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


    # Schmidt decomposition
    svd_u, singular_values, svd_v = _svd(state_vector)

    rank, svd_u, svd_v, singular_values = \
    _low_rank_approximation(low_rank, svd_u, svd_v, singular_values)

    circuit, reg_a, reg_b = _create_quantum_circuit(state_vector)

    # Phase 1. Encodes the singular values.
    if rank != 1:
        size_sv = len(singular_values)
        ebits = int(np.log2(size_sv))
        reg_sv = reg_b[:ebits]

        _encode(singular_values.reshape(size_sv, 1), circuit, reg_sv,
                                            isometry_scheme, unitary_scheme)

    # Phase 2. Entangles only the necessary qubits, according to rank.
    for j in range(int( np.log2( rank ) )):
        circuit.cx(reg_b[j], reg_a[j])

    # Phase 3 and 4 encode gates U and V.T
    _encode(svd_u, circuit, reg_b, isometry_scheme, unitary_scheme)
    _encode(svd_v.T, circuit, reg_a, isometry_scheme, unitary_scheme)
    
    return circuit


def _svd(state_vector):
    state = np.copy(state_vector)

    n_qubits = int(np.log2(len(state)))
    odd = n_qubits % 2
    lines = int(2**(n_qubits//2))
    cols = int(2**(n_qubits//2 + odd))
    state.shape = (lines, cols)

    svd_u, singular_values, svd_v = np.linalg.svd(state)
    singular_values = singular_values / np.linalg.norm(singular_values)

    return svd_u, singular_values, svd_v


def _low_rank_approximation(low_rank, svd_u, svd_v, singular_values):
    rank = svd_u.shape[0] # max rank
    effective_rank = sum(j > 10**-15 for j in singular_values)

    if 0 < low_rank < rank or effective_rank < rank:
        if 0 < low_rank < effective_rank:
            effective_rank = low_rank

        # To use isometries, the rank needs to be a power of 2.
        rank = int(2**np.ceil(np.log2(effective_rank)))
        svd_u = svd_u[:,:rank]
        svd_v = svd_v[:rank,:]
        singular_values = singular_values[:rank]

        singular_values = singular_values / np.linalg.norm(singular_values)

    return rank, svd_u, svd_v, singular_values


def _create_quantum_circuit(state):
    n_qubits = int(np.log2(len(state)))
    odd = n_qubits % 2
    reg_a = QuantumRegister(n_qubits//2 + odd)
    reg_b = QuantumRegister(n_qubits//2)
    circuit = QuantumCircuit(reg_a, reg_b)

    return circuit, reg_a, reg_b


def _encode(data, circuit, reg, iso_scheme='ccd', uni_scheme='qsd'):
    """
    Encodes data using the most appropriate method.
    """

    n_qubits = len(reg)
    _, svals, _ = _svd(data[:, 0])
    rank = sum(j > 10 ** -15 for j in svals)

    if data.shape[1] == 1 and (n_qubits % 2 == 0 or n_qubits < 4 or rank==1):

        # state preparation
        if n_qubits > 1:
            gate_u = initialize(
                data[:, 0],
                isometry_scheme=iso_scheme,
                unitary_scheme=uni_scheme)
        else:
            gate_u = mottonen(data[:, 0])

    elif data.shape[0] > data.shape[1]:
        gate_u = decompose_isometry(data, scheme=iso_scheme)
    else:
        gate_u = decompose_unitary(data, decomposition=uni_scheme)

    # Apply gate U to the register reg
    circuit.compose(gate_u, reg, inplace=True)



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

    if len(singular_values) > 2:
        circ = initialize_original(singular_values)
    else:
        circ = mottonen(singular_values)
    circuit.compose(circ, reg_b, inplace=True)

    for k in range(int(n_qubits//2)):
        circuit.cx(reg_b[k], reg_a[k])

    gate_u = decompose_unitary(svd_u, 'qsd')
    gate_v = decompose_unitary(svd_v.T, 'qsd')

    circuit.compose(gate_u, reg_b, inplace=True) # apply gate U to the first register
    circuit.compose(gate_v, reg_a, inplace=True) # apply gate V to the second register

    return circuit
