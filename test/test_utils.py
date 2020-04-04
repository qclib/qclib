"""
    This module is dedicated to implementing the tests in
    the qclib.utils module
"""
import unittest
from unittest import TestCase
import numpy as np
from qclib.utils import transform_dataset


class TestUtils(TestCase):
    """
        Class which implements the tests concerning
        the procedures in the qclib.utils module
    """

    def test_transform_dataset_unidimensional_list(self):
        """
            Testing  output of transform_dataset for a unidimentional
            list of values
        :return: None
        """

        input_vector = [-np.sqrt(0.03),
                        np.sqrt(0.02),
                        np.sqrt(0.02),
                        np.sqrt(0.03),
                        np.sqrt(0.1),
                        np.sqrt(0.4),
                        -np.sqrt(0.3),
                        -np.sqrt(0.1)]
        expected = [[(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')]]

        transformed = transform_dataset(input_vector)

        for returned_fv, expected_fv in zip(expected, transformed):
            for returned_feature, expected_feature in zip(returned_fv, expected_fv):
                test_real = returned_feature[0] == expected_feature[0]
                test_binary_pattern = returned_feature[1] == expected_feature[1]
                self.assertTrue(test_real and test_binary_pattern)

    def test_transform_dataset_bidimensional_list(self):
        """
            Testing  output of transform_dataset for a bidimentional
            list of values
        :return: None
        """
        input_vector = [[-np.sqrt(0.03),
                         np.sqrt(0.02),
                         np.sqrt(0.02),
                         np.sqrt(0.03),
                         np.sqrt(0.1),
                         np.sqrt(0.4),
                         -np.sqrt(0.3),
                         -np.sqrt(0.1)],
                        [-np.sqrt(0.03),
                         np.sqrt(0.02),
                         np.sqrt(0.02),
                         np.sqrt(0.03),
                         np.sqrt(0.1),
                         np.sqrt(0.4),
                         -np.sqrt(0.3),
                         -np.sqrt(0.1)]]

        expected = [[(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')],
                    [(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')]]

        transformed = transform_dataset(input_vector)

        for returned_fv, expected_fv in zip(expected, transformed):
            for returned_feature, expected_feature in zip(returned_fv, expected_fv):
                test_real = returned_feature[0] == expected_feature[0]
                test_binary_pattern = returned_feature[1] == expected_feature[1]
                self.assertTrue(test_real and test_binary_pattern)

    def test_transform_dataset_unidimensional_ndarray(self):
        """
            Testing  output of transform_dataset for a unidimentional
            numpy ndarray of values
        :return: None
        """

        input_vector = np.array([-np.sqrt(0.03),
                                 np.sqrt(0.02),
                                 np.sqrt(0.02),
                                 np.sqrt(0.03),
                                 np.sqrt(0.1),
                                 np.sqrt(0.4),
                                 -np.sqrt(0.3),
                                 -np.sqrt(0.1)])
        expected = [[(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')]]

        transformed = transform_dataset(input_vector)

        for returned_fv, expected_fv in zip(expected, transformed):
            for returned_feature, expected_feature in zip(returned_fv, expected_fv):
                test_real = returned_feature[0] == expected_feature[0]
                test_binary_pattern = returned_feature[1] == expected_feature[1]
                self.assertTrue(test_real and test_binary_pattern)

    def test_transform_dataset_bidimensional_ndarray(self):
        """
            Testing  output of transform_dataset for a bi-dimensional
            numpy ndarray of values
        :return: None
        """
        input_vector = np.array([[-np.sqrt(0.03),
                                  np.sqrt(0.02),
                                  np.sqrt(0.02),
                                  np.sqrt(0.03),
                                  np.sqrt(0.1),
                                  np.sqrt(0.4),
                                  -np.sqrt(0.3),
                                  -np.sqrt(0.1)],
                                 [-np.sqrt(0.03),
                                  np.sqrt(0.02),
                                  np.sqrt(0.02),
                                  np.sqrt(0.03),
                                  np.sqrt(0.1),
                                  np.sqrt(0.4),
                                  -np.sqrt(0.3),
                                  -np.sqrt(0.1)]])

        expected = [[(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')],
                    [(-np.sqrt(0.03), '000'),
                     (np.sqrt(0.02), '001'),
                     (np.sqrt(0.02), '010'),
                     (np.sqrt(0.03), '011'),
                     (np.sqrt(0.1), '100'),
                     (np.sqrt(0.4), '101'),
                     (-np.sqrt(0.3), '110'),
                     (-np.sqrt(0.1), '111')]]

        transformed = transform_dataset(input_vector)

        for returned_fv, expected_fv in zip(expected, transformed):
            for returned_feature, expected_feature in zip(returned_fv, expected_fv):
                test_real = returned_feature[0] == expected_feature[0]
                test_binary_pattern = returned_feature[1] == expected_feature[1]
                self.assertTrue(test_real and test_binary_pattern)

    def test_transform_dataset_basis_state(self):
        """
            Testing the preprocessing of an array which is in the
            basis state

        :return:None
        """
        input_vector = [0,
                        0,
                        0,
                        0,
                        0,
                        1,
                        0,
                        0]
        expected = [[(1, '101')]]

        transformed = transform_dataset(input_vector)

        for returned_fv, expected_fv in zip(expected, transformed):
            for returned_feature, expected_feature in zip(returned_fv, expected_fv):
                test_real = returned_feature[0] == expected_feature[0]
                test_binary_pattern = returned_feature[1] == expected_feature[1]
                self.assertTrue(test_real and test_binary_pattern)


if __name__ == '__main__':
    unittest.main()
