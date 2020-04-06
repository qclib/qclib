"""
 This module contains a set of unit tests dedicated to verify if the creation of quantum circuit
 in the Park's quantum circuit work properly.
"""
from unittest import TestCase
import qclib.encoding as encoding
import numpy as np
from qclib.utils import transform_dataset


class TestParksCircuit(TestCase):
    """
    Class dedicated to the implementation of the unittests for the creation of the
    quantum circuit that implements the Park's method
    """

    def test_dataset_with_one_feature_vector(self):
        """
            Test the output of the quantum circuit with a dataset with
            only one feature vector.
        :return: None
        """
        input_vector = np.array([np.sqrt(0.03),
                                 np.sqrt(0.47),
                                 np.sqrt(0.18),
                                 np.sqrt(0.32)])

        transformed_data, n_data_qbits = transform_dataset(input_vector)

        circuit = encoding.park_quantum_circuit(transformed_data, n_data_qbits)

        returned_state_vector = encoding.parks_state_vector_post_selection(circuit)

        expected_state_vector = [0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 np.sqrt(0.03), np.sqrt(0.47), np.sqrt(0.18), np.sqrt(0.32),
                                 0, 0, 0, 0]

        for returned, target in zip(returned_state_vector, expected_state_vector):
            self.assertTrue((returned - target) < 10**-5)

    def test_dataset_with_negative_valued_feature_vector(self):
        """
            Test the output of the quantum circuit with a dataset with
            only one feature vector with negative values
        :return: None
        """
        input_vector = -np.array([np.sqrt(0.03),
                                  np.sqrt(0.47),
                                  np.sqrt(0.18),
                                  np.sqrt(0.32)])

        transformed_data, n_data_qbits = transform_dataset(input_vector)

        circuit = encoding.park_quantum_circuit(transformed_data, n_data_qbits)

        returned_state_vector = encoding.parks_state_vector_post_selection(circuit)

        expected_state_vector = np.array([0, 0,
                                          0, 0,
                                          0, 0,
                                          0, 0,
                                          np.sqrt(0.03), np.sqrt(0.47),
                                          np.sqrt(0.18), np.sqrt(0.32),
                                          0, 0,
                                          0, 0])

        for returned, target in zip(returned_state_vector, expected_state_vector):
            self.assertTrue((returned - target) < 10**-5)

    def test_dataset_with_two_feature_vectors(self):
        """
            Test the output of the quantum circuit with a dataset with
            two feature vectors.
        :return: None
        """
        input_vector = np.array([[np.sqrt(0.03), np.sqrt(0.02), np.sqrt(0.02), np.sqrt(0.03)],
                                 [np.sqrt(0.1), np.sqrt(0.4), np.sqrt(0.3), np.sqrt(0.1)]])

        transformed_data, n_data_qbits = transform_dataset(input_vector)

        circuit = encoding.park_quantum_circuit(transformed_data, n_data_qbits)

        returned_state_vector = encoding.parks_state_vector_post_selection(circuit)

        expected_state_vector = [0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 np.sqrt(0.03), np.sqrt(0.02), np.sqrt(0.02), np.sqrt(0.03),
                                 np.sqrt(0.1), np.sqrt(0.4), np.sqrt(0.3), np.sqrt(0.1)]

        for returned, target in zip(returned_state_vector, expected_state_vector):
            self.assertTrue((returned - target) < 10 ** -5)

    def test_dataset_with_two_negative_valued_feature_vectors(self):
        """
            Test the output of the quantum circuit with a dataset with
            two negative valued feature vectors.
        :return: None
        """
        input_vector = -np.array([[np.sqrt(0.03), np.sqrt(0.02), np.sqrt(0.02), np.sqrt(0.03)],
                                  [np.sqrt(0.1), np.sqrt(0.4), np.sqrt(0.3), np.sqrt(0.1)]])

        transformed_data, n_data_qbits = transform_dataset(input_vector)

        circuit = encoding.park_quantum_circuit(transformed_data, n_data_qbits)

        returned_state_vector = encoding.parks_state_vector_post_selection(circuit)

        expected_state_vector = -np.array([0, 0,
                                           0, 0,
                                           0, 0,
                                           0, 0,
                                           np.sqrt(0.03), np.sqrt(0.02),
                                           np.sqrt(0.02), np.sqrt(0.03),
                                           np.sqrt(0.1), np.sqrt(0.4),
                                           np.sqrt(0.3), np.sqrt(0.1)])

        for returned, target in zip(returned_state_vector, expected_state_vector):
            self.assertTrue((returned - target) < 10 ** -5)

    def test_dataset_with_one_complex_valued_feature_vector(self):
        """
            Test the output of the quantum circuit
            for a dataset with one complex valued feature vector
        :return: None
        """
        input_vector = np.array([complex(np.sqrt(0.02), np.sqrt(0.01)),
                                 complex(np.sqrt(0.4), np.sqrt(0.07)),
                                 complex(np.sqrt(0.1), np.sqrt(0.08)),
                                 complex(np.sqrt(0.3), np.sqrt(0.02))])

        transformed_data, n_data_qbits = transform_dataset(input_vector)

        circuit = encoding.park_quantum_circuit(transformed_data, n_data_qbits)

        returned_state_vector = encoding.parks_state_vector_post_selection(circuit)

        expected_state_vector = [0, 0,
                                 0, 0,
                                 0, 0,
                                 0, 0,
                                 complex(np.sqrt(0.02), np.sqrt(0.01)), complex(np.sqrt(0.4), np.sqrt(0.07)),
                                 complex(np.sqrt(0.1), np.sqrt(0.08)), complex(np.sqrt(0.3), np.sqrt(0.02)),
                                 0, 0,
                                 0, 0]

        for returned, target in zip(returned_state_vector, expected_state_vector):
            self.assertTrue((returned - target) < 10 ** -5)
