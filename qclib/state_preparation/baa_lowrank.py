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
from qclib.state_preparation.initialize import Initialize
from qclib.state_preparation import LowRankInitialize
from qclib.state_preparation.util.baa import adaptive_approximation

class BaaLowRankInitialize(Initialize):
    """
    State preparation using the bounded approximation algorithm via Schmidt
    decomposition arXiv:1003.5760
    https://arxiv.org/abs/2111.03132

    This class implements a state preparation gate.
    """

    def __init__(self, params, inverse=False, label=None, opt_params=None):
        """
            Parameters
            ----------
            params: list of complex
                A unit vector representing a quantum state.
                Values are amplitudes.

            opt_params:
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

        """
        self._name = 'baa_lrsp'
        self._get_num_qubits(params)

        if opt_params is None:
            self.max_fidelity_loss = 0.0
            self.isometry_scheme = 'ccd'
            self.unitary_scheme = 'qsd'
            self.strategy='greedy'
            self.max_combination_size = 0
            self.use_low_rank = False
        else:
            if opt_params.get('max_fidelity_loss') is None:
                self.max_fidelity_loss = 0.0
            else:
                self.max_fidelity_loss = opt_params.get('max_fidelity_loss')

            if opt_params.get('iso_scheme') is None:
                self.isometry_scheme = 'ccd'
            else:
                self.isometry_scheme = opt_params.get('iso_scheme')

            if opt_params.get('unitary_scheme') is None:
                self.unitary_scheme = 'qsd'
            else:
                self.unitary_scheme = opt_params.get('unitary_scheme')

            if opt_params.get('strategy') is None:
                self.strategy = 'greedy'
            else:
                self.strategy = opt_params.get('strategy')

            if opt_params.get('max_combination_size') is None:
                self.max_combination_size = 0
            else:
                self.max_combination_size = opt_params.get('max_combination_size')

            if opt_params.get('use_low_rank') is None:
                self.use_low_rank = False
            else:
                self.use_low_rank = opt_params.get('use_low_rank')

            if opt_params.get('return_node') is None:
                self.return_node = False
            else:
                self.return_node = opt_params.get('return_node')

        if self.max_fidelity_loss < 0 or self.max_fidelity_loss > 1:
            self.max_fidelity_loss = 0.0

        self._label = label
        if label is None:
            self._label = 'SP'

            if inverse:
                self._label = 'SPdg'

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        node = adaptive_approximation(self.params, self.max_fidelity_loss, self.strategy,
                                        self.max_combination_size, self.use_low_rank)

        circuit = QuantumCircuit(self.num_qubits)

        for vector, qubits, rank, partition in zip(node.vectors, node.qubits,
                                                    node.ranks, node.partitions):

            lr_params = {'iso_scheme': self.isometry_scheme,
                         'unitary_scheme': self.unitary_scheme,
                         'partition': partition,
                         'lr': rank}

            gate = LowRankInitialize(vector, lr_params=lr_params)
            circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian.

        return circuit.reverse_bits()

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a BaaLowRankInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(BaaLowRankInitialize(state, opt_params=opt_params), q_circuit.qubits)
        else:
            q_circuit.append(BaaLowRankInitialize(state, opt_params=opt_params), qubits)


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

    node = adaptive_approximation(
        state_vector, max_fidelity_loss, strategy, max_combination_size, use_low_rank
    )

    n_qubits = int(np.log2(len(state_vector)))
    circuit = QuantumCircuit(n_qubits)

    for vector, qubits, rank, partition in zip(node.vectors, node.qubits,
                                                node.ranks, node.partitions):

        lr_params = {'iso_scheme': isometry_scheme,
                     'unitary_scheme': unitary_scheme,
                     'partition': partition,
                     'lr': rank}

        gate = LowRankInitialize(vector, lr_params=lr_params)
        circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian.

    if return_node:
        return circuit.reverse_bits(), node

    return circuit.reverse_bits()
