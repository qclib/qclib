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
    beta: float
    lmbda: float

    def __str__(self):
        return (
            f"level={self.level}_"
            f"index={self.index}\n"
            f"mag={self.mag:.2f}_"
            f"arg={self.arg:.2f}"
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
                cmath.phase(k.amplitude),
                0.0,
                0.0,
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

            # Calculates the substate.
            beta = 0.0
            if mag != 0.0:
                beta = nodes[k + 1].mag / mag

            lmbda = nodes[k + 1].arg - arg

            # Adds a new node to the tree.
            new_nodes.append(
                Node(
                    nodes[k].index // 2,
                    nqubits,
                    nodes[k],
                    nodes[k + 1],
                    mag,
                    arg,
                    beta,
                    lmbda,
                )
            )
            k = k + 2

    tree_root = new_nodes[0]
    return tree_root

def sparse_state_decomposition(nqubits, data):
    """
    :param nqubits: number of qubits required to generate a
                    state with the same length as the data vector (2^nqubits)
    :param data: list with pairs (index, amplitude)
    :return: root of the state tree
    """
    if len(data) == 0:
        return Node(0, 0, None, None, 0.0, 0.0, 0.0, 0.0)

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
                cmath.phase(k.amplitude),
                0.0,
                0.0,
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
            if nodes[k].index % 2 == 1:
                # Branching to the right.
                mag = nodes[k].mag
                arg = nodes[k].arg / 2

                # Adds a new node to the tree.
                new_nodes.append(
                    Node(
                        nodes[k].index // 2,
                        nqubits,
                        None,
                        nodes[k],
                        mag,
                        arg,
                        1.0,
                        arg,
                    )
                )
                k = k + 1
            elif (k + 1) < n_nodes and nodes[k + 1].index == nodes[k].index + 1:
                # Branching to the right and left.
                mag = math.sqrt(
                    nodes[k].mag ** 2 + nodes[k + 1].mag ** 2
                )
                arg = (
                    nodes[k].arg + nodes[k + 1].arg
                ) / 2

                # Calculates the substate.
                beta = 0.0
                if mag != 0.0:
                    beta = nodes[k + 1].mag / mag

                lmbda = nodes[k + 1].arg - arg

                # Adds a new node to the tree.
                new_nodes.append(
                    Node(
                        nodes[k].index // 2,
                        nqubits,
                        nodes[k],
                        nodes[k + 1],
                        mag,
                        arg,
                        beta,
                        lmbda,
                    )
                )
                k = k + 2
            else:
                # Branching to the left.
                mag = nodes[k].mag
                arg = nodes[k].arg / 2

                # Adds a new node to the tree.
                new_nodes.append(
                    Node(
                        nodes[k].index // 2,
                        nqubits,
                        nodes[k],
                        None,
                        mag,
                        arg,
                        0.0,
                        -arg,
                    )
                )
                k = k + 1

    tree_root = new_nodes[0]
    return tree_root
