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

from dataclasses import dataclass
from qiskit import QuantumCircuit
from qclib.gates.initialize import Initialize
from qclib.state_preparation.util.baa import adaptive_approximation
from .lowrank import LowRankInitialize


@dataclass
class _OptParams:
    def __init__(self, opt_params):
        if opt_params is None:
            self.max_fidelity_loss = 0.0
            self.isometry_scheme = "ccd"
            self.unitary_scheme = "qsd"
            self.strategy = "greedy"
            self.max_combination_size = 0
            self.use_low_rank = False
        else:
            self.max_fidelity_loss = 0.0 if opt_params.get("max_fidelity_loss") is None \
                else opt_params.get("max_fidelity_loss")

            self.isometry_scheme = "ccd" if opt_params.get("iso_scheme") is None else \
                opt_params.get("iso_scheme")

            self.unitary_scheme = "qsd" if opt_params.get("unitary_scheme") is None else \
                opt_params.get("unitary_scheme")

            self.strategy = "greedy" if opt_params.get("strategy") is None else \
                opt_params.get("strategy")

            self.max_combination_size = 0 if opt_params.get("max_combination_size") is None else \
                opt_params.get("max_combination_size")

            self.use_low_rank = False if opt_params.get("use_low_rank") is None else \
                opt_params.get("use_low_rank")

        if self.max_fidelity_loss < 0 or self.max_fidelity_loss > 1:
            self.max_fidelity_loss = 0.0

class BaaLowRankInitialize(Initialize):
    """
    State preparation using the bounded approximation algorithm via Schmidt
    decomposition arXiv:1003.5760
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

        opt_params: Dictionary
            max_fidelity_loss: float
                ``state`` allowed (fidelity) error for approximation
                (0<=``max_fidelity_loss``<=1). If ``max_fidelity_loss`` is not in the valid
                range, it will be ignored.

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
        self._name = "baa-lrsp"
        self._get_num_qubits(params)
        self.node = None
        self.opt_params = _OptParams(opt_params)

        if label is None:
            self._label = "BAASP"

        super().__init__(self._name, self.num_qubits, params, label=label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        self.node = adaptive_approximation(
            self.params,
            self.opt_params.max_fidelity_loss,
            self.opt_params.strategy,
            self.opt_params.max_combination_size,
            self.opt_params.use_low_rank,
        )

        circuit = QuantumCircuit(self.num_qubits)

        for vector, qubits, rank, partition in zip(
            self.node.vectors, self.node.qubits, self.node.ranks, self.node.partitions
        ):

            opt_params = {
                "iso_scheme": self.opt_params.isometry_scheme,
                "unitary_scheme": self.opt_params.unitary_scheme,
                "partition": partition,
                "lr": rank,
            }

            gate = LowRankInitialize(vector, opt_params=opt_params)
            circuit.compose(gate, qubits[::-1], inplace=True)  # qiskit little-endian.

        return circuit.reverse_bits()

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a BaaLowRankInitialize gate into the q_circuit
        """
        if qubits is None:
            q_circuit.append(
                BaaLowRankInitialize(state, opt_params=opt_params), q_circuit.qubits
            )
        else:
            q_circuit.append(BaaLowRankInitialize(state, opt_params=opt_params), qubits)
