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

    def __str__(self):
        space = '\t' * self.level
        txt = f"{space * self.level} y {self.angle_y:.2f} z{self.angle_z:.2f}\n"
        if self.left is not None:
            txt += self.left.__str__()
            txt += self.right.__str__()
        return txt


def create_angles_tree(state_tree):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    mag = 0.0
    if state_tree.mag != 0.0:
        mag = state_tree.right.mag / state_tree.mag

    arg = state_tree.right.arg - state_tree.arg

    # Avoid out-of-domain value due to numerical error.
    if mag < -1.0:
        angle_y = -math.pi
    elif mag > 1.0:
        angle_y = math.pi
    else:
        angle_y = 2 * math.asin(mag)

    angle_z = 2 * arg

    node = NodeAngleTree(
        state_tree.index, state_tree.level, angle_y, angle_z, None, None
    )

    if not is_leaf(state_tree.left):
        node.right = create_angles_tree(state_tree.right)
        node.left = create_angles_tree(state_tree.left)

    return node
