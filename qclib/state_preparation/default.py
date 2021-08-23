"""
	Default qclib state initialization functions.
	For dense state vectors, use ``initialize``.
	For sparse states, use ``sparse_initialize``.
"""

import numpy as np
import qiskit
from qclib.state_preparation.schmidt import initialize as dense_init
from qclib.state_preparation.sparse_isometry import initialize as sparse_init

# pylint: disable=maybe-no-member


def initialize(state):
    return dense_init(state)


def sparse_initialize(state):
	return sparse_init(state)
