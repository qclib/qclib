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


def _apply_controlled_rotations(quantum_circuit, controls, angle, n_qubits, with_barriers=False):
    """
    This procedure applies controlled rotations using a tuple
    with the states of the qubits (eg.: controls = (x, y, z),
    where each x,y and z can be either 0 or 1).
    A not gate is applied to a control qubit everytime it's in the state |0> ,
    given the controlled rotations are activated when the control qubit is set to |1>.

    :param quantum_circuit: Quantum circuit object from qiskit library
    :param controls: A tuple with one possibility of states of qubits, example: (0,1,0)
    :param angle: Angle to be used in the multi controlled rotation
    :param n_qubits: (int) Number of qubits in the quantum circtuin
    :param with_barriers: Boolean, add the barriers in the quantum circuit for better visualisation
                          when printing the circuit
    :return: Updated quantum circuit with the controlled rotations applied
    """

    n_controls = len(controls)

    control_qubit_indexes = list(range(n_controls))

    for i, ctrl in enumerate(controls):
        if ctrl == 0:
            quantum_circuit.x(n_qubits - i - 1)

    if with_barriers:
        quantum_circuit.barrier()
    # Applying controlled rotation with using the angle,
    # the indexes of control qubits and its target qubit

    control_qubit_objects_list = []
    reg = quantum_circuit.qregs[0]
    for c_idx in control_qubit_indexes:
        control_qubit_objects_list.append(reg[n_qubits - c_idx - 1])

    target = reg[n_qubits - n_controls - 1]
    quantum_circuit.mcry(angle, control_qubit_objects_list, target, None, mode='noancilla')

    if with_barriers:
        quantum_circuit.barrier()

    for i, ctrl in enumerate(controls):
        if ctrl == 0:
            quantum_circuit.x(n_qubits - i - 1)

    return quantum_circuit


def _check_angles_list(angles):
    """
    Check if the poperties of the input are according to the type of input expected in the
    cascading_ry procedure

    :param angles: input angles
    :return: adjusted input angles
    """

    if isinstance(angles, list) and angles == []:
        raise Exception("Empty iterable")

    if isinstance(angles[0], (list, tuple, np.ndarray)):
        raise Exception("Expected an uni-dimensional iterable")

    if isinstance(angles, np.ndarray):
        angles = angles.tolist()

    return angles


def _cascading_ry(angles, with_barriers=False):
    """
    This procedure creates the quantum circuit for the Mottonen method for phase encoding.
    Building a coherent superposition from the ground state with the features encoded in the phase.

    :param angles: list of values to be coded in the phases
    :param with_barriers: Boolean, add the barriers in the quantum circuit for better visualisation
                        when printing the circuit
    :return: The quantum circuit with the gates
    """

    angles = _check_angles_list(angles)

    n_qubits = int(np.ceil(np.log2(len(angles) + 1)))

    q_cirq = QuantumCircuit(n_qubits)

    # Building Circuit
    current_value = angles.pop(0)

    q_cirq.ry(current_value, n_qubits - 1)# pylint: disable=maybe-no-member

    for i in range(1, n_qubits):

        # Creates a list with tuples of all combinations of binary strings with size i
        c_qubits = list(product([0, 1], repeat=i))

        for controls in c_qubits:
            if with_barriers:
                q_cirq.barrier() # pylint: disable=maybe-no-member
            current_value = angles.pop(0)
            q_cirq = _apply_controlled_rotations(q_cirq, controls, current_value,
                                                 n_qubits, with_barriers=with_barriers)

    return q_cirq


def _resize_feature_vectors(features):
    """
    Resize the feature vector if its dimension is not in a 2^n format
    by concatenating zeroes.
    The input is expected to be uni-dimensional.

    :param features: Feature vector to be resized
    :return: feature vector
    """

    features_size = len(features)
    if np.log2(features_size) < np.ceil(np.log2(features_size)):

        multiplier = int(2 ** np.ceil(np.log2(features_size)) - features_size)
        features = np.concatenate((features, [0] * multiplier))
    return features


def mottonen_quantum_circuit(features, with_barriers=False):
    """
        Generates the quantum circuit for the Mottonen's method based on a feature vector of
        real numbers.
    :param features: The feature vector to be encoded in the quantum state,
                     it is expected to be normalized.
    :param with_barriers: Boolean, add the barriers in the quantum circuit for better visualisation
                        when printing the circuit
    :return: Quantum Circuit object generated to perform Mottonen's method
    """

    features_norm = np.linalg.norm(features)

    if np.ceil(features_norm) != 1:
        features_norm = features_norm / features_norm

    features = _resize_feature_vectors(features)

    angles = []
    _recursive_compute_angles(features, angles)

    return _cascading_ry(angles, with_barriers=with_barriers)
