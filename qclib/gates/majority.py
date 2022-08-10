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

""" Majority gate """

from itertools import combinations
import numpy as np


def operate(circuit, controls, target):
    """Apply a majority gate"""

    size_controls = len(controls)
    log_n = int(np.floor(np.log2(size_controls)))

    n_min = int(np.ceil(size_controls / 2))
    n_max = 2**log_n

    n_controls = [n_min]

    if n_min != n_max:
        if n_min % 2 != 0:
            n_controls.extend(range(n_min + 1, n_max))

        n_controls.append(n_max)

    for k in n_controls:
        comb = combinations(controls, k)
        for j in comb:
            circuit.mcx([*j], target)
