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
Bounded Approximation version of Plesch's algorithm.
https://arxiv.org/abs/2111.03132
"""

from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from qclib.state_preparation.util.baa import adaptive_approximation, Node
from qclib.state_preparation import schmidt

def initialize(state_vector, low_rank=0, max_fidelity_loss=0.0,
                    isometry_scheme='ccd', unitary_scheme='qsd'):
    """
    State preparation using the bounded approximation algorithm via Schmidt
    decomposition arXiv:1003.5760

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize(state)

    Parameters
    ----------
    state_vector: list of float or array-like
        A unit vector representing a quantum state.
        Values are amplitudes.

    low_rank: int
        ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
        If ``low_rank`` is not in the valid range, it will be ignored.
        If the parameters ``low_rank`` and ``max_fidelity_loss`` are used simultaneously,
        the fidelity of the final state may be less than 1-``max_fidelity_error`` .

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

    if max_fidelity_loss < 0 or max_fidelity_loss > 1:
        max_fidelity_loss = 0.0

    node_option: Optional[Node] = adaptive_approximation(state_vector, max_fidelity_loss)
    if node_option is None:
        return schmidt.initialize(state_vector, low_rank=low_rank, isometry_scheme=isometry_scheme,
                                                                   unitary_scheme=unitary_scheme)

    n_qubits = int(np.ceil(np.log2(len(state_vector))))
    circuit = QuantumCircuit(n_qubits)
    offset = 0
    for vec in node_option.vectors:
        vec_n_qubits = int(np.ceil(np.log2(len(vec))))
        if vec_n_qubits == 1:
            qc_vec = QuantumCircuit(1)
            qc_vec.initialize(vec) # pylint: disable=no-member
        else:
            qc_vec = schmidt.initialize(vec, low_rank=low_rank, isometry_scheme=isometry_scheme,
                                                                unitary_scheme=unitary_scheme)
        affected_qubits = list(range(offset, offset + vec_n_qubits))
        circuit = circuit.compose(qc_vec, affected_qubits)
        offset += vec_n_qubits

    return circuit
