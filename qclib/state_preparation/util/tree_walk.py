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
https://arxiv.org/abs/2108.10182
"""

from qiskit.circuit.library import RYGate, RZGate
from qclib.gates.ucr import ucr
from qclib.state_preparation.util.tree_utils import leftmost, children

def bottom_up(angle_tree, circuit, start_level):
    """ bottom_up state preparation """

    if angle_tree and angle_tree.level < start_level:

        if angle_tree.angle_y != 0.0:
            circuit.ry(angle_tree.angle_y, angle_tree.qubit)
            if angle_tree.angle_z != 0.0:
                circuit.rz(angle_tree.angle_z, angle_tree.qubit)

        bottom_up(angle_tree.left, circuit, start_level)
        bottom_up(angle_tree.right, circuit, start_level)

        _apply_cswaps(angle_tree, circuit)

def top_down(angle_tree, circuit, start_level, control_nodes=None, target_nodes=None):
    """ top down state preparation """
    if angle_tree:
        if angle_tree.level < start_level:
            top_down(angle_tree.left, circuit, start_level)
            top_down(angle_tree.right, circuit, start_level)
        else:
            if target_nodes is None:
                control_nodes = []                           # initialize the controls
                target_nodes = [angle_tree]                  # start by the subtree root
            else:
                target_nodes = children(target_nodes)        # all the nodes in the current level

            angles_y = [node.angle_y for node in target_nodes]
            angles_z = [node.angle_z for node in target_nodes]
            target_qubit = target_nodes[0].qubit
            control_qubits = [node.qubit for node in control_nodes]

            # If both multiplexors are used (RY and RZ), we can save two CNOTs.
            # That is why the RZ multiplexor is reversed.
            if any(angles_y):
                ucry = ucr(RYGate, angles_y, last_control=not any(angles_z))
                circuit.append(ucry, [target_qubit]+control_qubits[::-1])

            if any(angles_z):
                ucrz = ucr(RZGate, angles_z, last_control=not any(angles_y))
                circuit.append(ucrz.reverse_ops(), [target_qubit]+control_qubits[::-1])

            control_nodes.append(angle_tree)                 # add current node to the controls list

            # walk to the first node of the next level.
            top_down(angle_tree.left, circuit, start_level,
                     control_nodes=control_nodes, target_nodes=target_nodes)

def _apply_cswaps(angle_tree, circuit):

    if angle_tree.angle_y != 0.0:
        left = angle_tree.left
        right = angle_tree.right

        while left and right:
            circuit.cswap(angle_tree.qubit, left.qubit, right.qubit)

            left = left.left
            right = leftmost(right)
