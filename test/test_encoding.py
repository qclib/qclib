"""
 This module contains a set of unit tests dedicated to verify if the creation of quantum circuit
 in the mottonen module work properly.
"""
import unittest
from unittest import TestCase
from random import randint
import numpy as np
from qiskit import execute, Aer
import qclib.encoding as encoding
from qclib.utils import transform_dataset


class TestMottonenCircuit(TestCase):
    """
    Class dedicated to the implementation of the unittests for the creation of the quantum circuit.
    """

    def test_mottonen_initialization(self):
        """
        Load a 3 qubits state
        """
        input_vector = [-np.sqrt(0.03),
                        np.sqrt(0.02),
                        np.sqrt(0.02),
                        np.sqrt(0.03),
                        np.sqrt(0.1),
                        np.sqrt(0.4),
                        -np.sqrt(0.3),
                        -np.sqrt(0.1)]
        quantum_circuit = encoding.mottonen_quantum_circuit(input_vector)
        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)

        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_mottonen_initialization_basis_state(self):
        """
        Load a basis state
        """
        input_vector = [0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0]
        quantum_circuit = encoding.mottonen_quantum_circuit(input_vector)
        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)

        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_mottonen_initialization_random_4qubtis(self):
        """
        exemplo de teste
        :return:
        """
        input_vector = np.random.rand(16)
        input_vector = input_vector / np.linalg.norm(input_vector)
        quantum_circuit = encoding.mottonen_quantum_circuit(input_vector)
        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)

        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_mottonen_random_sized_feature_vector(self):
        """
        Testing mottonen's quantum circuit resulting state vector for a randomly sized feature
        vector
        :return:
        """

        input_vector = np.random.uniform(low=-10, high=10, size=randint(2, 17))
        input_vector = input_vector / np.linalg.norm(input_vector)
        input_vector_resized = encoding._resize_feature_vectors(input_vector)
        quantum_circuit = encoding.mottonen_quantum_circuit(input_vector)
        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector()
        for exp_amplitude, out_amplitude in zip(input_vector_resized, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))


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


if __name__ == "__main__":
    unittest.main()
