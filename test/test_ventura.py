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
from qclib.state_preparation.ventura import initialize
from qclib.util import get_state

class TestInitialize(TestCase):
        
    def test_binary_function(self):
        """
          Testing a function (f(z) \in {0, 1, ..., N-1}) with z \in {0,1}^n.
        """
        a = {0:0, 1:1, 2:2, 3:3, 4:4, 11:0, 12:1, 13:2, 14:3, 15:4} # couples of input and output.
        m = len(a)
        N = max(a.values())-1
        n = 4
        
        circuit = initialize(a, n=n, N=N)
        state = get_state(circuit)
        
        b = []
        for k in range(2**n):
            if (k in a): # The amplitudes of modulus "1/sqrt(m)" will be "2 pi / N" radians apart from each other on the complex plane.
                b.append(-1/np.sqrt(m)*np.exp( a[k]*1j*2*np.pi/N ) )
            else:
                b.append(0)
        
        self.assertTrue(np.allclose(b, state[:2**n])) # The data is encoded in the first 2^n amplitudes of the circuit state vector.
    