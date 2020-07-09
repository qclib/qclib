"""
 This module contains a set of unit tests dedicated to verify if the creation of quantum circuit
 in the mottonen module work properly.
"""
from unittest import TestCase
from random import randint
import numpy as np
from qiskit import execute, Aer, QuantumRegister
from qclib import QuantumCircuit
from qclib.encoding import InitializerUniformlyRotation


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

        quantum_register = QuantumRegister(3)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.ur_initialize(input_vector, quantum_register)

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

        quantum_register = QuantumRegister(3)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.ur_initialize(input_vector, quantum_register)

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

        quantum_register = QuantumRegister(4)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.ur_initialize(input_vector, quantum_register)

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

        quantum_register = QuantumRegister(size)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.ur_initialize(input_vector, quantum_register)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector()
        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_mult_initialization_random_4qubtis(self):
        """
        Load a 4 qubit state
        """
        input_vector = np.random.rand(16)
        input_vector = input_vector / np.linalg.norm(input_vector)

        quantum_register = QuantumRegister(4)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.mult_initialize(input_vector, quantum_register)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()

        out_state = result.get_statevector(quantum_circuit)
        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))

    def test_phase_extraction(self):
        """
            Test phase extraction procedure for phase encoding
        """
        expected_phases = np.array([0.9553166181245093, 0.3962075716952029,
                                    0.7297276562269663, 0.25268025514207865])

        complex_vector = [complex(np.sqrt(0.01), np.sqrt(0.02)),
                          complex(np.sqrt(0.4), np.sqrt(0.07)),
                          complex(np.sqrt(0.1), np.sqrt(0.08)),
                          complex(np.sqrt(0.3), np.sqrt(0.02))]

        _, returned_phases = InitializerUniformlyRotation._extract_phase_from_complex(0, complex_vector)

        for phase_index in range(len(expected_phases)):
            self.assertTrue(np.abs(expected_phases[phase_index] - returned_phases[phase_index]) < 10e-5)

    def test_phase_with_only_one_entry(self):
        """
            Testing phase calculation where the state vector has only one entry
        """

        expected_phases = [0.7853981633974483, 0.0, 0.0, 0.0]
        input_vector = np.array([complex(np.sqrt(0.5), np.sqrt(0.5)), 0, 0, 0])

        _, returned_phases = InitializerUniformlyRotation._extract_phase_from_complex(0, input_vector)

        for phase_index in range(len(expected_phases)):
            self.assertTrue(np.abs(expected_phases[phase_index] - returned_phases[phase_index]) < 10e-5)

    def test_phase_angles_computation(self):
        """
            Testing phases calculations of the phase encoding procedure
        """

        complex_vector = [complex(np.sqrt(0.01), np.sqrt(0.02)),
                          complex(np.sqrt(0.4), np.sqrt(0.07)),
                          complex(np.sqrt(0.1), np.sqrt(0.08)),
                          complex(np.sqrt(0.3), np.sqrt(0.02))]

        phases_vector = np.array([0.9553166181245093,
                                  0.3962075716952029,
                                  0.7297276562269663,
                                  0.25268025514207865])

        expected_angles = np.array([-0.0922790696126668,
                                    -0.27955452321465324,
                                    -0.23852370054244385])

        initializer = InitializerUniformlyRotation(complex_vector)
        initializer._compute_phase_equalization_angles(phases_vector)
        phases_angles = initializer._angles_tree

        for angles_indexes in range(len(phases_angles)):
            self.assertTrue(np.abs(expected_angles[angles_indexes] - phases_angles[angles_indexes]) < 10e-5)

    def test_phase_angles_computations_with_one_entry(self):
        """
            Test phases calculations for feature vectors with only one entry
        """
        input_vector = [complex(np.sqrt(0.5), np.sqrt(0.5)), 0, 0, 0]
        phases_vector = [0.7853981633974483, 0.0, 0.0, 0.0]

        expected_angles = [-0.19634954084936207, -0.39269908169872414, 0.0]

        initializer = InitializerUniformlyRotation(input_vector)
        initializer._compute_phase_equalization_angles(phases_vector)
        phases_angles = initializer._angles_tree

        for angles_indexes in range(len(phases_angles)):
            self.assertTrue(np.abs(expected_angles[angles_indexes] - phases_angles[angles_indexes]) < 10e-5)

    def test_global_phase_application(self):
        """
            Test  the output of the circuit if only the global phase is applied
        """
        mock_params = [complex(np.sqrt(0.01), np.sqrt(0.02)),
                       complex(np.sqrt(0.4), np.sqrt(0.07)),
                       complex(np.sqrt(0.1), np.sqrt(0.08)),
                       complex(np.sqrt(0.3), np.sqrt(0.02))]
        
        phases_vector = [0.9553166181245093,
                         0.3962075716952029,
                         0.7297276562269663,
                         0.25268025514207865]

        expected_state = [0.834548798785615+0.5509340273077049j, 0.0, 0.0, 0.0]

        expected_global_phase = 0.5834830252971893

        global_phase = 1 / 4 * np.sum(phases_vector)

        self.assertTrue(np.abs(expected_global_phase - global_phase) < 10e-5)

        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        initializer = InitializerUniformlyRotation(mock_params)
        initializer.num_qubits = 2
        initializer._apply_global_phase(phases_vector)

        backend = Aer.get_backend('statevector_simulator')
        job = execute(qc, backend)
        result = job.result()
        returned_state = result.get_statevector()

        for angles_indexes in range(len(expected_state)):
            self.assertTrue(np.abs(expected_state[angles_indexes] - returned_state[angles_indexes]) < 10e-5)

    def test_ur_intialization_with_phase_encoding(self):
        
        input_vector = [complex(np.sqrt(0.01), np.sqrt(0.02)),
                        complex(np.sqrt(0.4), np.sqrt(0.07)),
                        complex(np.sqrt(0.1), np.sqrt(0.08)),
                        complex(np.sqrt(0.3), np.sqrt(0.02))]

        quantum_register = QuantumRegister(2)
        quantum_circuit = QuantumCircuit(quantum_register)
        quantum_circuit.ur_initialize(input_vector, quantum_register)

        backend_sim = Aer.backends('statevector_simulator')[0]
        job = execute(quantum_circuit, backend_sim)
        result = job.result()
        out_state = result.get_statevector()
        for exp_amplitude, out_amplitude in zip(input_vector, out_state):
            self.assertTrue(np.abs(exp_amplitude - out_amplitude) < 10 ** (-5))
