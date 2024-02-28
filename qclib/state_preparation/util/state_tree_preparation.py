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
import cmath
from dataclasses import dataclass
from typing import NamedTuple


class Amplitude(NamedTuple):
    """
    Named tuple for amplitudes
    """

    index: int
    amplitude: float

    def __str__(self):
        return f"{self.index}:{self.amplitude:.2f}"


@dataclass
class Node:
    """
    Binary tree node used in state_decomposition function
    """

    index: int
    level: int
    left: "Node"
    right: "Node"
    mag: float
    arg: float

    def __str__(self):
        return (
            f"{self.level}_"
            f"{self.index}\n"
            f"{self.mag:.2f}_"
            f"{self.arg:.2f}"
        )


def state_decomposition(nqubits, data):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with exactly 2^nqubits pairs (index, amplitude)
    :return: root of the state tree
    """
    new_nodes = []

    # leafs
    for k in data:
        new_nodes.append(
            Node(
                k.index,
                nqubits,
                None,
                None,
                abs(k.amplitude),
                cmath.phase(k.amplitude)
            )
        )

    # build state tree
    while nqubits > 0:
        nodes = new_nodes
        new_nodes = []
        nqubits = nqubits - 1
        k = 0
        n_nodes = len(nodes)
        while k < n_nodes:
            mag = math.sqrt(
                nodes[k].mag ** 2 + nodes[k + 1].mag ** 2
            )
            arg = (
                nodes[k].arg + nodes[k + 1].arg
            ) / 2

            new_nodes.append(
                Node(nodes[k].index // 2, nqubits, nodes[k], nodes[k + 1], mag, arg)
            )
            k = k + 2

    tree_root = new_nodes[0]
    return tree_root
