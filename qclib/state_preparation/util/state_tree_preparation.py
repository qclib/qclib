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
        return str(self.index) + ':' + \
               f'{self.amplitude:.2g}'

@dataclass
class Node:
    """
    Binary tree node used in state_decomposition function
    """

    index: int
    level: int
    amplitude: float
    left: 'Node'
    right: 'Node'

    def __str__(self):
        return str(self.level) + '_' + \
               str(self.index) + '\n' + \
               f'{self.amplitude:.2g}'

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
        new_nodes.append(Node(k.index, nqubits, k.amplitude, None, None))

    # build state tree
    while nqubits > 0:
        nodes = new_nodes
        new_nodes = []
        nqubits = nqubits - 1
        k = 0
        n_nodes = len(nodes)
        while k < n_nodes:
            mag = math.sqrt(abs(nodes[k].amplitude) ** 2 + abs(nodes[k + 1].amplitude) ** 2)
            arg = (cmath.phase(nodes[k].amplitude) + cmath.phase(nodes[k + 1].amplitude)) / 2

            amp = mag * cmath.exp(1j*arg)

            new_nodes.append(Node(nodes[k].index // 2, nqubits,
                                  amp,
                                  nodes[k],
                                  nodes[k + 1]))
            k = k + 2

    tree_root = new_nodes[0]
    return tree_root
