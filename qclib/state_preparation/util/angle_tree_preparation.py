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

import cmath
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
        return (
            str(self.level)
            + "_"
            + str(self.index)
            + "\n"
            + f"{self.angle_y:.2g}_"
            + f"{self.angle_z:.2g}"
        )


def create_angles_tree(state_tree):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    amp = 0.0
    if state_tree.amplitude != 0.0:
        amp = state_tree.right.amplitude / state_tree.amplitude

    # Avoid out-of-domain value due to numerical error.
    if abs(amp) < -1.0:
        angle_y = -math.pi
    elif abs(amp) > 1.0:
        angle_y = math.pi
    else:
        angle_y = 2 * math.asin(abs(amp))

    angle_z = 2 * cmath.phase(amp)

    node = NodeAngleTree(
        state_tree.index, state_tree.level, angle_y, angle_z, None, None
    )

    if not is_leaf(state_tree.left):
        node.right = create_angles_tree(state_tree.right)
        node.left = create_angles_tree(state_tree.left)

    return node
