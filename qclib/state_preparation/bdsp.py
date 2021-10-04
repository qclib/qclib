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

from math import ceil
from qiskit import QuantumCircuit

from qclib.state_preparation.util.state_tree_preparation import *
from qclib.state_preparation.util.angle_tree_preparation import *
from qclib.state_preparation.util.tree_register          import *
from qclib.state_preparation.util.tree_walk              import top_down, bottom_up

def initialize(state, split=None):
    """
        https://arxiv.org/abs/2108.10182
    """
    n_qubits = int(np.log2(len(state)))
    data = [Amplitude(i, a) for i, a in enumerate(state)]

    state_tree = state_decomposition(n_qubits, data)
    angle_tree = create_angles_tree(state_tree)
    
    if (split == None):
        split = int(ceil(n_qubits/2)) # sublinear

    circuit = QuantumCircuit()
    add_register(circuit, angle_tree, n_qubits-split)

    top_down(angle_tree, circuit, n_qubits-split)
    bottom_up(angle_tree, circuit, n_qubits-split)
    
    return circuit

