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

