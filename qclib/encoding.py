"""
 Loading real vectors in the amplitude of a quantum system based on arXiv:quant-ph/0407010v1
"""
from itertools import product
import numpy as np
from qiskit import QuantumCircuit


def _recursive_compute_angles(input_vector, angles_tree):
    """
    :param input_vector: The feature vector to be encoded in the quantum state,
                         it is expected to be normalized.
    :param angles_tree: empty list to store the angles
    :return:
    """
    if len(input_vector) > 1:
        new_input = []
        angles = []
        for k in range(0, len(input_vector), 2):
            norm = np.sqrt(input_vector[k] ** 2 + input_vector[k + 1] ** 2)
            new_input.append(norm)
            if norm == 0:
                angles.append(0)
            else:
                if input_vector[k] < 0:
                    angles.append(2 * np.pi - 2 * np.arcsin(input_vector[k + 1] / norm))
                else:
                    angles.append(2 * np.arcsin(input_vector[k + 1] / norm))
        _recursive_compute_angles(new_input, angles_tree)
        for value in angles:
            angles_tree.append(value)