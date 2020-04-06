"""
    Module dedicated to running the test cases implemented
    for the encoding algorithms
"""
import unittest
from test_mottonnen import TestMottonenCircuit
from test_park import TestParksCircuit

if __name__ == '__main__':

    test_clases = [TestMottonenCircuit, TestParksCircuit]

    loader = unittest.TestLoader()

    suite_list = []
    for test_class in test_clases:
        suite = loader.loadTestsFromTestCase(test_class)
        suite_list.append(suite)

    b_suite = unittest.TestSuite(suite_list)
    runner = unittest.TextTestRunner()
    runner.run(b_suite)
