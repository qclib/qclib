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

import math
from dataclasses import dataclass
from qclib.state_preparation.util.tree_utils import is_leaf
from qclib.state_preparation.util.state_tree_preparation import Node

@dataclass
class NodeAngleTree:
    """
    Binary tree node used in function create_angles_tree
    """

    index: int
    level: int
    angle_y: float
    angle_z: float
    left: "NodeAngleTree"
    right: "NodeAngleTree"
    parent: "NodeAngleTree"
    state_node: "Node"
    pruned: bool = False

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.angle_y:.2f}_"
            f"{self.angle_z:.2f}"
        )


def create_angles_tree(state_tree, parent=None):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    beta  = state_tree.beta
    lmbda  = state_tree.lmbda

    # Avoid out-of-domain value due to numerical error.
    if beta < -1.0:
        angle_y = -math.pi
    elif beta > 1.0:
        angle_y = math.pi
    else:
        angle_y = 2 * math.asin(beta)

    angle_z = 2 * lmbda

    node = NodeAngleTree(
        state_tree.index,
        state_tree.level,
        angle_y,
        angle_z,
        None,
        None,
        parent,
        state_tree
    )

    if state_tree.left and not is_leaf(state_tree.left):
        node.left = create_angles_tree(state_tree.left, node)

    if state_tree.right and not is_leaf(state_tree.right):
        node.right = create_angles_tree(state_tree.right, node)

    return node
