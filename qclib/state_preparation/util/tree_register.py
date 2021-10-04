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

import qiskit
from qclib.state_preparation.util.tree_utils import children

"""
https://arxiv.org/abs/2108.10182
"""

def output(angle_tree, q):
    if angle_tree:
        q.insert(0, angle_tree.qubit) # qiskit little-endian
        if angle_tree.left:
            output(angle_tree.left, q)
        else:
            output(angle_tree.right, q)

def _add_register(angle_tree, qubits, start_level):
    if angle_tree:
        angle_tree.qubit = qubits.pop(0)
        if angle_tree.level < start_level:
            _add_register(angle_tree.left, qubits, start_level)
            _add_register(angle_tree.right, qubits, start_level)
        else:
            if angle_tree.left:
                _add_register(angle_tree.left, qubits, start_level)
            else:
                _add_register(angle_tree.right, qubits, start_level)

def add_register(circuit, angle_tree, start_level):
    """
    Organize qubit registers, grouping by "output" and "ancilla" types.
    """
    
    level = 0
    level_nodes = []
    nodes = [angle_tree]
    while len(nodes) > 0: # count nodes per level
        level_nodes.append( len(nodes) )
        nodes = children(nodes)
        level += 1

    noutput = level # one output qubits per level
    nqubits = sum(level_nodes[:start_level]) # bottom-up qubits
    nqubits += level_nodes[start_level] * (noutput-start_level) # top-down qubits: (number of sub-states) * (number of qubits per sub-state)
    nancilla = nqubits - noutput

    output_register = qiskit.QuantumRegister(noutput, name='output')
    circuit.add_register(output_register)
    qubits = [*output_register[::-1]]

    if nancilla > 0:
        ancilla_register = qiskit.QuantumRegister(nancilla, name='ancilla')
        circuit.add_register(ancilla_register)
        qubits.extend([*ancilla_register[::-1]])

    _add_register(angle_tree, qubits, start_level)
