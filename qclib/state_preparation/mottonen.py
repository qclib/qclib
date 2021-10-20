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
        https://arxiv.org/abs/quant-ph/0407010
        https://arxiv.org/abs/2108.10182
"""

import numpy as np
from qiskit import QuantumCircuit

from qclib.state_preparation.util.state_tree_preparation import Amplitude, state_decomposition

from qclib.state_preparation.util.angle_tree_preparation import \
    create_angles_tree

from qclib.state_preparation.util.tree_register import add_register
from qclib.state_preparation.util.tree_walk import top_down


def initialize(state, global_phase=True):
    """
        https://arxiv.org/abs/quant-ph/0407010
        https://arxiv.org/abs/2108.10182
    """
    n_qubits = int(np.log2(len(state)))
    data = [Amplitude(i, a) for i, a in enumerate(state)]

    state_tree = state_decomposition(n_qubits, data)
    angle_tree = create_angles_tree(state_tree)

    circuit = QuantumCircuit()
    add_register(circuit, angle_tree, 0)

    top_down(angle_tree, circuit, 0)
    if global_phase:
        # equivalent to unitary(I * exp(1j*sum(np.angle(state))/len(state)))
        circuit.global_phase += sum(np.angle(state))/len(state)

    return circuit
