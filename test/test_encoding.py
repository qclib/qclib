"""
 This module contains a set of unit tests dedicated to verify if the creation of quantum circuit
 in the mottonen module work properly.
"""
import unittest
from unittest import TestCase
from random import randint
import numpy as np
from qiskit import execute, Aer, QuantumRegister
from qclib import QuantumCircuit


class TestCircuitCreation(TestCase):
    """
    Class dedicated to the implementation of the unittests for the creation of the quantum circuit.
    """

    def test_ur_initialization(self):
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

        qr = QuantumRegister(3)
        quantum_circuit = QuantumCircuit(qr)
        quantum_circuit.ur_initialize(input_vector, qr)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)

        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_ur_initialization_basis_state(self):
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

        qr = QuantumRegister(3)
        quantum_circuit = QuantumCircuit(qr)
        quantum_circuit.ur_initialize(input_vector, qr)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)

        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_ur_initialization_random_4qubtis(self):
        """
        Load a 4 qubit state
        """
        input_vector = np.random.rand(16)
        input_vector = input_vector / np.linalg.norm(input_vector)

        qr = QuantumRegister(4)
        quantum_circuit = QuantumCircuit(qr)
        quantum_circuit.ur_initialize(input_vector, qr)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)
        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_ur_random_sized_feature_vector(self):
        """
        Testing mottonen's quantum circuit resulting state vector for a randomly sized feature
        vector
        :return:
        """

        size = randint(2, 5)
        input_vector = np.random.uniform(low=-10, high=10, size=2**size)
        input_vector = input_vector / np.linalg.norm(input_vector)

        qr = QuantumRegister(size)
        quantum_circuit = QuantumCircuit(qr)
        quantum_circuit.ur_initialize(input_vector, qr)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector()
        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

if __name__ == "__main__":
    unittest.main()
