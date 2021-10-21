""" Test sparse initialization """

from unittest import TestCase
import numpy as np
from qclib.state_preparation.cvoqram import cvoqram_initialize
from qclib.util import double_sparse, get_state


class TestCvoqram(TestCase):
    """ Testing qclib.state_preparation.cvoqram """
    def test_cvoqram(self):
        """ Testing cvoqram 2 amplitudes and with auxiliary qubits """
        data = [([0, 0, 1], 1/np.sqrt(3)), ([1, 1, 0], np.sqrt(2j/3))]
        qc_cvoqram = cvoqram_initialize(data)
        state = get_state(qc_cvoqram)
        self.assertTrue(np.isclose(state[0b110000], np.sqrt(2j/3)))
        self.assertTrue(np.isclose(state[0b001000], np.sqrt(1/3)))

    def test_cvoqram_without_aux(self):
        """ Testing cvoqram 2 amplitudes and with auxiliary qubits """
        data = [([0, 0, 1], 1j * np.sqrt(1/3)), ([1, 1, 0], np.sqrt(2j/3))]
        qc_cvoqram = cvoqram_initialize(data, False)
        state = get_state(qc_cvoqram)
        self.assertTrue(np.isclose(state[0b1100], np.sqrt(2j/3)))
        self.assertTrue(np.isclose(state[0b0010], 1j * np.sqrt(1/3)))

    def test_cvoqram_random(self):
        """ Testing cvoqram 4 amplitudes and with auxiliary qubits """
        n_qubits = 4
        log_npatterns = 2
        prob = 0.2
        data = double_sparse(n_qubits, log_npatterns, prob)
        qc_cvoqram = cvoqram_initialize(data)
        state = get_state(qc_cvoqram)
        for k, _ in enumerate(data):
            lst = data[k][0]
            bin_index = ''.join(map(str, lst))
            bin_index = bin_index + n_qubits * '0'  # padding work qubits

            self.assertTrue(np.isclose(state[int(bin_index, 2)], data[k][1]))

    def test_double_sparse(self):
        """ Test double sparse random generation """
        n_qubits = 6
        log_npatterns = 4
        prob = 0.8
        data = double_sparse(n_qubits, log_npatterns, prob)
        norm = 0
        for k, _ in enumerate(data):
            norm = norm + np.abs(data[k][1])**2
            self.assertTrue(np.sum(data[k][0]) <= np.sum(data[k][0]))

        self.assertTrue(np.isclose(norm, 1))
