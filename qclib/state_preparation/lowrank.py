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
from qiskit import QuantumCircuit
from qclib.unitary import unitary as decompose_unitary, cnot_count as cnots_unitary
from qclib.isometry import decompose as decompose_isometry, cnot_count as cnots_isometry
from qclib.gates.initialize import Initialize
from qclib.entanglement import schmidt_decomposition, _to_qubits
from .topdown import TopDownInitialize

# pylint: disable=maybe-no-member


class LowRankInitialize(Initialize):
    """
    Approximated quantum-state preparation with entanglement dependent complexity
    https://arxiv.org/abs/2111.03132

    This class implements a state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of complex
            A unit vector representing a quantum state.
            Values are amplitudes.

        opt_params: {'lr': low_rank,
                     'iso_scheme': isometry_scheme,
                     'unitary_scheme': unitary_scheme,
                     'partition': partition}
            low_rank: int
                ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
                If ``low_rank`` is not in the valid range, it will be ignored.
                This parameter limits the rank of the Schmidt decomposition. If the Schmidt rank
                of the state decomposition is greater than ``low_rank``, a low-rank
                approximation is applied.

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

            svd: string
                Function to compute the SVD, acceptable values are 'auto' (default), 'regular',
                and 'randomized'. 'auto' sets `svd='randomized'` for `n_qubits>=12 and rank==1`.
        """
        self._name = "low_rank"
        self._get_num_qubits(params)

        if opt_params is None:
            self.isometry_scheme = "ccd"
            self.unitary_scheme = "qsd"
            self.low_rank = 0
            self.partition = None
            self.svd = "auto"
        else:
            self.low_rank = 0 if opt_params.get("lr") is None else opt_params.get("lr")
            self.partition = opt_params.get("partition")
            if opt_params.get("iso_scheme") is None:
                self.isometry_scheme = "ccd"
            else:
                self.isometry_scheme = opt_params.get("iso_scheme")

            if opt_params.get("unitary_scheme") is None:
                self.unitary_scheme = "qsd"
            else:
                self.unitary_scheme = opt_params.get("unitary_scheme")

            if opt_params.get("svd") is None:
                self.svd = "auto"
            else:
                self.svd = opt_params.get("svd")


        if label is None:
            label = "LRSP"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        if self.num_qubits < 2:
            return TopDownInitialize(self.params).definition

        circuit, reg_a, reg_b = self._create_quantum_circuit()

        # Schmidt decomposition
        rank, svd_u, singular_values, svd_v = schmidt_decomposition(
            self.params, reg_a, rank=self.low_rank, svd=self.svd
        )

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
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a LowRankInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                LowRankInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(LowRankInitialize(state, opt_params=opt_params), qubits)

    def _encode(self, data, circuit, reg):
        """
        Encodes data using the most appropriate method.
        """
        if data.shape[1] == 1:
            # state preparation
            gate_u = LowRankInitialize(data[:, 0], opt_params={
                "iso_scheme": self.isometry_scheme,
                "unitary_scheme": self.unitary_scheme,
                "svd": self.svd
            })

        elif data.shape[0] // 2 == data.shape[1]:
            # isometry 2^(n-1) to 2^n.
            gate_u = decompose_isometry(data, scheme="csd")

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


def _default_partition(n_qubits):
    odd = n_qubits % 2
    return list(range(n_qubits // 2 + odd))


def cnot_count(
    state_vector,
    low_rank=0,
    isometry_scheme="ccd",
    unitary_scheme="qsd",
    partition=None,
    method = "estimate",
    svd="auto"
):
    """
    Estimate the number of CNOTs to build the state preparation circuit.
    """

    n_qubits = _to_qubits(len(state_vector))
    if n_qubits < 2:
        return 0

    if partition is None:
        partition = _default_partition(n_qubits)

    cnots = 0

    rank, svd_u, singular_values, svd_v = schmidt_decomposition(
        state_vector,
        partition,
        rank=low_rank,
        svd=svd
    )

    # Schmidt measure of entanglement
    ebits = _to_qubits(rank)

    # Phase 1.
    if ebits > 0:
        singular_values = singular_values / np.linalg.norm(singular_values)
        cnots += _cnots(
            singular_values.reshape(rank, 1), isometry_scheme, unitary_scheme, method, svd
        )
    # Phase 2.
    cnots += ebits

    # Phases 3 and 4.
    cnots += _cnots(svd_u, isometry_scheme, unitary_scheme, method, svd)
    cnots += _cnots(svd_v.T, isometry_scheme, unitary_scheme, method, svd)

    return cnots


def _cnots(data, iso_scheme="ccd", uni_scheme="qsd", method="estimate", svd="auto"):
    if data.shape[1] == 1:
        return cnot_count(
            data[:, 0],
            isometry_scheme=iso_scheme,
            unitary_scheme=uni_scheme,
            method=method,
            svd=svd
        )

    if data.shape[0] // 2 == data.shape[1]:
        return cnots_isometry(data, scheme="csd", method=method)

    if data.shape[0] > data.shape[1]:
        return cnots_isometry(data, scheme=iso_scheme, method=method)

    return cnots_unitary(data, decomposition=uni_scheme, method=method)
