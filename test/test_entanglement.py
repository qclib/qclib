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

""" Test for entanglement.py module"""

from unittest import TestCase
import numpy as np
from qclib.entanglement import geometric_entanglement

class TestEntanglement(TestCase):
    """ Tests for entanglement.py"""
    def test_geometric_measure_entanglement(self):
        """ Test geometric measure of entanglement """
        ghz4 = np.zeros(16)
        ghz4[0] = 1 / np.sqrt(2)
        ghz4[15] = 1 / np.sqrt(2)
        gme = geometric_entanglement(ghz4)
        self.assertTrue(np.isclose(gme, 0.5))

        nqbits = 5
        w6_state = np.zeros(2**nqbits)
        for k in range(nqbits):
            w6_state[2**k] = 1 / np.sqrt(nqbits)

        gme = geometric_entanglement(w6_state)
        self.assertTrue(np.abs(gme - 0.59) < 1e-3)

    def test_geometric_positive(self):
        """Test if geometric entanglemen is positive"""
        vector = np.array(
            [
                0.00186779 + 0.01306012j,
                -0.01104266 - 0.07721341j,
                -0.01704493 + 0.00199258j,
                0.10077226 - 0.01178041j,
                -0.00188656 - 0.01597529j,
                0.01115364 + 0.09444834j,
                0.01555059 + 0.00254013j,
                -0.09193749 - 0.01501764j,
                -0.00979644 - 0.06849949j,
                0.05791804 + 0.40497947j,
                0.0893996 - 0.01045093j,
                -0.52854411 + 0.06178752j,
                0.00989489 + 0.08378937j,
                -0.05850013 - 0.49537556j,
                -0.08156188 - 0.01332283j,
                0.48220629 + 0.07876658j,
            ]
        )
        gme = geometric_entanglement(vector.tolist())
        self.assertTrue(gme > 0)
