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

import numpy as np
from qiskit import QuantumCircuit
from qclib.state_preparation import schmidt
from qclib.state_preparation.util.baa import adaptive_approximation

def initialize(state_vector, max_fidelity_loss=0.0,
                        isometry_scheme='ccd', unitary_scheme='qsd',
                        strategy='greedy', max_combination_size=0,
                        use_low_rank=False, return_node=False):
    """
    State preparation using the bounded approximation algorithm via Schmidt
    decomposition arXiv:1003.5760

    For instance, to initialize the state a|00> + b|10> (|a|^2+|b|^2=1)
        $ state = [a, 0, b, 0]
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

    strategy: string
        Method to search for the best approximation (``'brute_force'`` or ``'greedy'``).
        For states larger than 2**8, the greedy strategy should preferably be used.
        Default is ``strategy='greedy'``.

    max_combination_size: int
        Maximum size of the combination ``C(n_qubits, max_combination_size)``
        between the qubits of an entangled subsystem of length ``n_qubits`` to
        produce the possible bipartitions
        (1 <= ``max_combination_size`` <= ``n_qubits``//2).
        For example, if ``max_combination_size``==1, there will be ``n_qubits``
        bipartitions between 1 and ``n_qubits``-1 qubits.
        The default value is 0 (the size will be maximum for each level).

    use_low_rank: bool
            If set to True, ``rank``>1 approximations are also considered. This is fine
            tuning for high-entanglement states and is slower.
            The default value is False.

    return_node: bool
            If set to true, returns also the best node for the
            decomposition/approximation

    Returns
    -------
    circuit: QuantumCircuit or Tuple[QuantumCircuit, Node]
        QuantumCircuit to initialize the state or if return_node==True, returns
        a tuple with the QuantumCircuit and the best node for
        decomposition/approximation.
    """

    if max_fidelity_loss < 0 or max_fidelity_loss > 1:
        max_fidelity_loss = 0.0

    # Completely separates the state to compute the maximum possible fidelity loss.
    # With the below parameter setting, the cost of the function "adaptive_approximation"
    # is linear in the number of qubits. If max_fidelity_loss input is less than the
    # maximum possible loss, it runs the full routine with potentially exponential cost.
    node = adaptive_approximation(state_vector, 1.0, strategy='greedy', max_combination_size=1)
    if (node.total_fidelity_loss) > max_fidelity_loss:
        node = adaptive_approximation(state_vector, max_fidelity_loss, strategy,
                                                max_combination_size, use_low_rank)

    n_qubits = int(np.log2(len(state_vector)))
    circuit = QuantumCircuit(n_qubits)

    for i, vec in enumerate(node.vectors):
        qc_vec = schmidt.initialize(vec, isometry_scheme=isometry_scheme,
                                            unitary_scheme=unitary_scheme)
        circuit.compose(qc_vec, node.qubits[i][::-1], inplace=True) # qiskit little-endian.

    if return_node:
        return circuit.reverse_bits(), node

    return circuit.reverse_bits()
