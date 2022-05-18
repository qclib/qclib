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
from qiskit import QuantumCircuit
from qclib.state_preparation.mottonen import initialize as mottonen
from qclib.unitary import unitary as decompose_unitary, cnot_count as cnots_unitary
from qclib.isometry import decompose as decompose_isometry, cnot_count as cnots_isometry
from qclib.state_preparation.initialize import Initialize
from qclib.entanglement import schmidt_decomposition, _to_qubits, _effective_rank

# pylint: disable=maybe-no-member


class LowRankInitialize(Initialize):
    """
    Low-rank state preparation
    https://arxiv.org/abs/2111.03132

    This class implements a state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, lr_params=None):
        """
            Parameters
            ----------
            params: list of complex
                A unit vector representing a quantum state.
                Values are amplitudes.

            lr_params: {'lr': low_rank,
                        'iso_scheme': isometry_scheme,
                        'unitary_scheme': unitary_scheme,
                        'partition': partition}
                low_rank: int
                    ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
                    If ``low_rank`` is not in the valid range, it will be ignored.
                    This parameter limits the rank of the Schmidt decomposition. If the Schmidt rank
                    of the state decomposition is greater than ``low_rank``, a low-rank approximation
                    is applied.

                iso_scheme: string
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

                """
        self._name = 'low_rank'
        self._get_num_qubits(params)

        if lr_params is None:
            self.isometry_scheme = 'ccd'
            self.unitary_scheme = 'qsd'
            self.low_rank = 0
            self.partition = None
        else:
            self.low_rank = 0 if lr_params.get('lr') is None else lr_params.get('lr')
            self.partition = lr_params.get('partition')
            if lr_params.get('iso_scheme') is None:
                self.isometry_scheme = 'ccd'
            else:
                self.isometry_scheme = lr_params.get('iso_scheme')

            if lr_params.get('unitary_scheme') is None:
                self.unitary_scheme = 'qsd'
            else:
                self.unitary_scheme = lr_params.get('unitary_scheme')

        self._label = label
        if label is None:
            self._label = 'SP'

            if inverse:
                self._label = 'SPdg'

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        if self.num_qubits < 2:
            return mottonen(self.params)

        circuit, reg_a, reg_b = self._create_quantum_circuit()

        # Schmidt decomposition
        svd_u, singular_values, svd_v = schmidt_decomposition(self.params, reg_a)

        rank, svd_u, svd_v, singular_values = \
            low_rank_approximation(self.low_rank, svd_u, svd_v, singular_values)

        # Schmidt measure of entanglement
        e_bits = _to_qubits(rank)

        # Phase 1. Encodes the singular values.
        if e_bits > 0:
            reg_sv = reg_b[:e_bits]
            singular_values = singular_values / np.linalg.norm(singular_values)
            self._encode(singular_values.reshape(rank, 1), circuit, reg_sv)

        # Phase 2. Entangles only the necessary qubits, according to rank.
        for j in range(e_bits):
            circuit.cx(reg_b[j], reg_a[j])

        # Phase 3 and 4 encode gates U and V.T
        self._encode(svd_u, circuit, reg_b)
        self._encode(svd_v.T, circuit, reg_a)

        return circuit.reverse_bits()

    @staticmethod
    def initialize(q_circuit, state, qubits=None, lr_params=None):
        """
        Appends a LowRankInitialize gate into the q_circuit
        """
        if lr_params is None:
            lr_params = {'lr': 0, 'iso_scheme': 'ccd', 'unitary_scheme': 'qsd', 'partition': None}
        if qubits is None:
            q_circuit.append(LowRankInitialize(state, lr_params=lr_params), q_circuit.qubits)
        else:
            q_circuit.append(LowRankInitialize(state, lr_params=lr_params), qubits)

    def _encode(self, data, circuit, reg):
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
            lr_params = {'iso_scheme': self.isometry_scheme,
                         'unitary_scheme': self.unitary_scheme}
            gate_u = LowRankInitialize(data[:, 0], lr_params=lr_params)

        elif data.shape[0] > data.shape[1]:
            gate_u = decompose_isometry(data, scheme=self.isometry_scheme)
        else:
            gate_u = decompose_unitary(data, decomposition=self.unitary_scheme)

        # Apply gate U to the register reg
        circuit.compose(gate_u, reg, inplace=True)

    def _create_quantum_circuit(self):

        if self.partition is None:
            self.partition = _default_partition(self.num_qubits)

        complement = sorted(set(range(self.num_qubits)).difference(set(self.partition)))

        circuit = QuantumCircuit(self.num_qubits)

        return circuit, self.partition[::-1], complement[::-1]


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


def _default_partition(n_qubits):
    odd = n_qubits % 2
    return list(range(n_qubits//2 + odd))


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
