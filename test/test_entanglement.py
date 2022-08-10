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

from unittest import TestCase
import numpy as np
from qclib.entanglement import geometric_entanglement
from qclib.state_preparation.util.baa import adaptive_approximation


class TestEntanglement(TestCase):
    def test_geometric_measure_entanglement(self):
        ghz4 = np.zeros(16)
        ghz4[0] = 1 / np.sqrt(2)
        ghz4[15] = 1 / np.sqrt(2)
        gme = geometric_entanglement(ghz4)
        self.assertTrue(np.isclose(gme, 0.5))

        nqbits = 5
        w6 = np.zeros(2**nqbits)
        for k in range(nqbits):
            w6[2**k] = 1 / np.sqrt(nqbits)

        gme = geometric_entanglement(w6)
        self.assertTrue(np.abs(gme - 0.59) < 1e-3)
