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

# import numpy as np
from math import isclose
from qiskit.circuit.library import RYGate, RZGate
from qclib.gates.ucr import ucr
from qclib.gates import Mcg
from qclib.gates.mcg import cnot_count
from qclib.state_preparation.util.tree_utils import leftmost, children, subtree_level_index

def bottom_up(angle_tree, circuit, start_level):
    """bottom_up state preparation"""

    if angle_tree and angle_tree.level < start_level:

        if angle_tree.qubit:
            if angle_tree.angle_y != 0.0:
                circuit.ry(angle_tree.angle_y, angle_tree.qubit)
            if angle_tree.angle_z != 0.0:
                circuit.rz(angle_tree.angle_z, angle_tree.qubit)

        bottom_up(angle_tree.left, circuit, start_level)
        bottom_up(angle_tree.right, circuit, start_level)

        _apply_cswaps(angle_tree, circuit)


def top_down(angle_tree, circuit, start_level, control_nodes=None, target_nodes=None):
    """top down state preparation"""
    if angle_tree:
        if angle_tree.level < start_level: # search sub-trees at split level.
            top_down(angle_tree.left, circuit, start_level)
            top_down(angle_tree.right, circuit, start_level)
        else:
            if target_nodes is None:
                control_nodes = []  # initialize the controls
                target_nodes = [angle_tree]  # start by the subtree root
            else:
                target_nodes = children(
                    target_nodes
                )  # all the nodes in the current level

            target_qubit = target_nodes[0].qubit

            if target_qubit:
                k_nodes = len(control_nodes)

                angles_y = [node.angle_y for node in target_nodes if not isclose(node.angle_y, 0.0)]
                angles_z = [node.angle_z for node in target_nodes if not isclose(node.angle_z, 0.0)]

                if k_nodes == 0: # no controls
                    if any(angles_y):
                        circuit.ry(angles_y[0], target_qubit)
                    if any(angles_z):
                        circuit.rz(angles_z[0], target_qubit)
                else:
                    control_qubits = [node.qubit for node in control_nodes]
                    ucr_y = False
                    ucr_z = False

                    root = control_nodes[0]

                    # Collects node's control information.
                    for target_node in target_nodes:
                        index = subtree_level_index(root, target_node)

                        ctrl_state = f"{index:0{k_nodes}b}"

                        ctrl_nodes = [target_node.parent]
                        while ctrl_nodes[-1].parent and ctrl_nodes[-1].parent.level >= root.level:
                            ctrl_nodes.append(ctrl_nodes[-1].parent)
                        ctrl_nodes = ctrl_nodes[::-1]

                        target_node.subtree_index = index
                        target_node.ctrl_state = ''.join(
                            c for i, c in enumerate(ctrl_state)
                            if ctrl_nodes[i].left and ctrl_nodes[i].right
                        )
                        target_node.ctrl_qubits = [
                            c for i, c in enumerate(control_qubits)
                            if ctrl_nodes[i].left and ctrl_nodes[i].right
                        ]

                    # Pad with zeros for the case where the UCR is more
                    # efficient. The condition below should be adjusted
                    # according to the optimization of the multicontrolled.
                    mcg_cnot_count = 0
                    for target_node in target_nodes:
                        if not isclose(target_node.angle_y, 0.0):
                            mcg_cnot_count += cnot_count(
                                RYGate(target_node.angle_y).to_matrix(),
                                len(target_node.ctrl_state)
                            )

                    if 2**k_nodes < mcg_cnot_count:
                        ucr_y = any(angles_y)
                        angles_y = [0.0] * 2**k_nodes
                        for target_node in target_nodes:
                            if not isclose(target_node.angle_y, 0.0):
                                angles_y[target_node.subtree_index] = target_node.angle_y

                    mcg_cnot_count = 0
                    for target_node in target_nodes:
                        if not isclose(target_node.angle_z, 0.0):
                            mcg_cnot_count += cnot_count(
                                RZGate(target_node.angle_z).to_matrix(),
                                len(target_node.ctrl_state)
                            )

                    if 2**k_nodes < mcg_cnot_count:
                        ucr_z = any(angles_z)
                        angles_z = [0.0] * 2**k_nodes
                        for target_node in target_nodes:
                            if not isclose(target_node.angle_z, 0.0):
                                angles_z[target_node.subtree_index] = target_node.angle_z

                    # If both multiplexors are used (RY and RZ), we can save two CNOTs.
                    # That is why the RZ multiplexor is reversed.
                    if any(angles_y):
                        if ucr_y:
                            ucry = ucr(
                                RYGate,
                                angles_y,
                                last_control = not ucr_z
                            )
                            circuit.append(
                                ucry,
                                [target_qubit] + control_qubits[::-1]
                            )
                        else:
                            for target_node in target_nodes:
                                if not isclose(target_node.angle_y, 0.0):
                                    if len(target_node.ctrl_qubits) == 0:
                                        circuit.ry(target_node.angle_y, target_qubit)
                                    else:
                                        mcgate = Mcg(
                                            RYGate(target_node.angle_y).to_matrix(),
                                            len(target_node.ctrl_state),
                                            target_node.ctrl_state
                                        )
                                        circuit.append(
                                            mcgate,
                                            target_node.ctrl_qubits[::-1] + [target_qubit]
                                        )

                    if any(angles_z):
                        if ucr_z:
                            ucrz = ucr(
                                RZGate,
                                angles_z,
                                last_control = not ucr_y
                            )
                            circuit.append(
                                ucrz.reverse_ops(),
                                [target_qubit] + control_qubits[::-1]
                            )
                        else:
                            for target_node in target_nodes:
                                if not isclose(target_node.angle_z, 0.0):
                                    if len(target_node.ctrl_qubits) == 0:
                                        circuit.rz(target_node.angle_z, target_qubit)
                                    else:
                                        mcgate = Mcg(
                                            RZGate(target_node.angle_z).to_matrix(),
                                            len(target_node.ctrl_state),
                                            target_node.ctrl_state
                                        )
                                        circuit.append(
                                            mcgate,
                                            target_node.ctrl_qubits[::-1] + [target_qubit]
                                        )

                control_nodes.append(angle_tree)  # add current node to the controls list

            # walk to the first node of the next level.
            top_down(
                leftmost(angle_tree),
                circuit,
                start_level,
                control_nodes=control_nodes,
                target_nodes=target_nodes,
            )


def _apply_cswaps(angle_tree, circuit):

    if angle_tree.angle_y != 0.0:
        left = angle_tree.left
        right = angle_tree.right

        while left and right:
            if not left.pruned and not right.pruned:
                circuit.cswap(angle_tree.qubit, left.qubit, right.qubit)

            left = leftmost(left)
            right = leftmost(right)
