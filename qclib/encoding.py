"""
 Loading real vectors in the amplitude of a quantum system based on arXiv:quant-ph/0407010v1
"""
from itertools import product
from abc import ABC, abstractmethod
import numpy as np
from qiskit.circuit import Instruction

from qclib import QuantumCircuit


class Initializer(ABC, Instruction):
    @abstractmethod
    def initialize(self, params, qubits):
        pass


class InitializerUniformlyRotation(Initializer):
    """
    State preparation arXiv:quant-ph/0407010
    """
    def __init__(self, params):
        """
        Encode a unit vector in a quantum state
        params (list): probability amplitudes
        """
        features_norm = np.linalg.norm(params)

        if not np.isclose(features_norm, 1):
            params = params / features_norm

        self._angles_tree = []
        self._phases_tree = []
        self._phase_vector = []
        self.params = params
        self.num_qubits = int(np.log2(len(params)))
        self._circuit = QuantumCircuit(self.num_qubits)

        super().__init__("initialize UR", self.num_qubits, 0, self.params)

    def _extract_phase_from_complex(self, input_vector):
        """
            Extract the phase of any complex entry in the feature vector
        :param input_vector: The feature vector to be encoded in the quantum state,
                             it is expected to be normalized.
        :return: The phase of the entries in the complex

        """
        phases = []

        for idx_entry, entry in enumerate(input_vector):
            if isinstance(entry, complex):
                phases += np.log(entry).imag
                input_vector[idx_entry] = np.linalg.norm(entry)
            else:
                phases += [0]
        return phases

    def _apply_global_phase(self, phase_vector):
        """
            Apply global phase to the most significant qubit.
            According to arXiv:quant-ph/0407010v1 after the phase equalization
            the global phase of the state is the mean of the all phase angles.
            That is

            exp(i * 1 / 2**n_qubits * sum(phase_vector))

            In order to compute the reversed process of phase equalization
            the quantum state must have the presented global phase


        :param phase_vector: list with the extracted phases from the complex features
        :return: None
        """
        omega = 1 / 2**self.num_qubits * np.sum(phase_vector)
        # most significant qubit
        ms_qubit = self.num_qubits - 1
        self._circuit.x(ms_qubit)
        self._circuit.u1(omega, ms_qubit)
        self._circuit.x(ms_qubit)
        self._circuit.u1(omega, ms_qubit)

    def _get_angle_computation(self, phase_vector, start, end, skip):
        """
        Performs two-by-two subtraction between the phase angles in the phase vector
        using skip to define which angles keep a positive signal and which keep a negative

        Let "phase_idx" be the phase index in the phase_vector. And let "a" be the angle

            a_(phase_idx + skip) - a_(phase_idx)

        :param phase_vector: list with the extracted phases from the complex features
        :param start: index of the starting angle of the two-by-two subtraction
        :param end: index of the ending angle of the two-by-two subtraction
        :param skip: the skiping index to define which term will be the positive term
                    the subtracion
        :return: sumation of a two-by-two subtraction of angles in the phase_vector
        """
        sumation = 0
        for phase_idx in range(start, end):
            sumation += (phase_vector[phase_idx + skip] - phase_vector[phase_idx])
        return sumation

    def _compute_phase_equalization_angles(self, phase_vector):
        """
        Extracts the phase from complex entries in the input vector
        Computes the phase equalization angles to be used when applying the cascading
        controlled Rz

        :param input_vector: The feature vector to be encoded in the quantum state,
                             it is expected to be normalized.
        :return: None
        """
        tree_level = 0
        phase_angles = []
        while tree_level < self.num_qubits:

            main_factor = 1 / 2 ** (self.num_qubits - tree_level)
            # Defining how many indices must be skiped before applying subtraction
            skip = 2 ** (self.num_qubits - tree_level - 1)
            # Defining start and  end indices
            start = 0
            end = 2 ** (self.num_qubits - tree_level - 1)
            # defining how many cells the start and end indices must be shifted
            offset = 2 ** (self.num_qubits - tree_level)
            n_nodes = 2 ** tree_level
            for node in range(n_nodes):
                phase_angles.append(main_factor * self._get_angle_computation(phase_vector, start, end, skip))
                start += offset
                end += offset

            tree_level += 1
        self._phases_tree = phase_angles

    def _recursive_compute_amplitude_angles(self, input_vector):
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
            self._recursive_compute_amplitude_angles(new_input)
            for value in angles:
                self._angles_tree.append(value)

    def _apply_controlled_rotations(self, controls, angle, phase, n_qubits):
        """
        This procedure applies controlled rotations using a tuple
        with the states of the qubits (eg.: controls = (x, y, z),
        where each x,y and z can be either 0 or 1).
        A not gate is applied to a control qubit everytime it's in the state |0> ,
        given the controlled rotations are activated when the control qubit is set to |1>.

        :param phase:
        :param controls: A tuple with one possibility of states of qubits, example: (0,1,0)
        :param angle: Angle to be used in the multi controlled rotation
        :param n_qubits: (int) Number of qubits in the quantum circtuin
        """

        # Define global phase
        self._apply_global_phase(self._phase_vector)

        n_controls = len(controls)
        control_qubit_indexes = list(range(n_controls))

        for i, ctrl in enumerate(controls):
            if ctrl == 0:
                self._circuit.x(n_qubits - i - 1)

        # Applying controlled rotation with using the angle,
        # the indexes of control qubits and its target qubit

        control_qubit_objects_list = []
        reg = self._circuit.qregs[0]
        for c_idx in control_qubit_indexes:
            control_qubit_objects_list.append(reg[n_qubits - c_idx - 1])

        target = reg[n_qubits - n_controls - 1]
        self._circuit.mcry(angle, control_qubit_objects_list, target, None, mode='noancilla')

        # If phase == 0 then the Rz becomes Identity,
        # thus increasing the depth of the circuit unnecessarily
        if phase != 0:
            self._circuit.mcrz(phase, control_qubit_objects_list, target)

        for i, ctrl in enumerate(controls):
            if ctrl == 0:
                self._circuit.x(n_qubits - i - 1)

    def _create_circuit(self):
        """
        This procedure creates the quantum circuit for the Mottonen method for phase encoding.
        Building a coherent superposition from the ground state with the features encoded in the phase.
        """
        angles = self._angles_tree
        phases = self._phases_tree

        n_qubits = int(np.ceil(np.log2(len(angles) + 1)))

        self._apply_global_phase(self._phase_vector)

        # Building Circuit
        current_value = angles.pop(0)
        current_phase = phases.pop(0)

        self._circuit.ry(current_value, n_qubits - 1)
        self._circuit.rz(current_phase, n_qubits - 1)

        for i in range(1, n_qubits):

            # Creates a list with tuples of all combinations of binary strings with size i
            c_qubits = list(product([0, 1], repeat=i))

            for controls in c_qubits:
                current_value = angles.pop(0)
                current_phase = phases.pop(0)
                self._apply_controlled_rotations(controls, current_value, current_phase, n_qubits)

    def _define(self):
        """
            Generates the quantum circuit for the Mottonen's method based on a feature vector of
            real numbers.
        :param features: The feature vector to be encoded in the quantum state,
                         it is expected to be normalized.
        :return: Quantum Circuit object generated to perform Mottonen's method
        """

        self._recursive_compute_amplitude_angles(self.params)
        self._phase_vector = self._extract_phase_from_complex(self.params)
        self._compute_phase_equalization_angles(self._phase_vector)
        self._create_circuit()
        self.definition = self._circuit.data

    def initialize(self, params, qubits):
        return self.append(InitializerUniformlyRotation(params), qubits)

    QuantumCircuit.ur_initialize = initialize


class InitializerMultiplexor(InitializerUniformlyRotation):
    """
    State preparation arXiv:quant-ph/0406176
    """
    def _create_circuit(self):
        self._circuit.ry(self._angles_tree[0], self.num_qubits-1)
        for k in range(1, self.num_qubits):
            angles = self._angles_tree[2 ** k - 1: 2 ** (k + 1) - 1]
            qubits = list(range(self.num_qubits-k-1, self.num_qubits))
            self._circuit.ry_multiplexor(angles, qubits)

    def initialize(self, params, qubits):
        return self.append(InitializerMultiplexor(params), qubits)

    QuantumCircuit.mult_initialize = initialize
