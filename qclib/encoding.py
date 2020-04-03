"""
 Loading real vectors in the amplitude of a quantum system based on arXiv:quant-ph/0407010v1
"""
from itertools import product
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


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


def _initialize_state(quantum_circuit, initialize_index=True):
    """
        Auxiliary procedure for the method of Park, et al, for initializing
        the states.
        If the dataset only has one feature vector then it is not necessary to
        apply data initialization to the ancila
    :param quantum_circuit: Quantum circuit object where Park's method will
                            be applied
    :param n_feature_qubits: Number of qubits needed for encoding the features in to the
                             quantum state
    :param n_dset_size_qubits: Number of qubits needed for encoding the index qubit
                               for each feature in the dataset
    :return: Quantum Circuit with state initialization implemented
    """

    quantum_data = quantum_circuit.qregs[0]
    quantum_index = quantum_circuit.qregs[2]

    for qb_idx in range(quantum_data.size):
        quantum_circuit.h(quantum_data[qb_idx])

    if initialize_index:
        for qb_idx in range(quantum_index.size):
            quantum_circuit.h(quantum_index[qb_idx])

    return quantum_circuit


def _qubitwise_not(binary_pattern, quantum_circuit, quantum_data):
    """
        Auxiliary Procedure that applies the flip/flop step in Park's method.
    :param binary_pattern: Binary pattern associated to the continuous value
    :param quantum_circuit: quantum circuit in which the flip step is going to be applied
    :param quantum_data: Quantum Register object for the qubits dedicated to the data
    :return: Quantum circuit updated with the flip step
    """
    for bit_index, _ in enumerate(binary_pattern):
        quantum_circuit.x(quantum_data[bit_index])
    return quantum_circuit


def _apply_apply_multi_controlled_rotation(angle, quantum_circuit, control_register,
                                           target_register, target_index,
                                           rotation_type='rz'):
    """
        Apply multicontrolled rotation to the quantum circuit using all quibits
        in the control_register parameter as control, and the target_register as target qubit,
        according to the target index.
    :param angle: Angle to be used in the rotation
    :param quantum_circuit: quantum circuit in which the rotation is going to be applied
    :param control_register: Quantum Register object in which all its qubits will be used as control
    :param target_register: Quantum Register object which contains the target qubit
    :param n_controls: Size of the control register
    :param target_index: Index of the target qubit of the rotation
    :param rotation_type: rotation type, allowing to chose among the
                          multicontrolled rotaions Ry, Rz and Rx
    :return: Quantum circuit updated with the R_*(angle) rotation applied
    """

    ctrl_qubits_list = []

    for i in range(control_register.size):
        ctrl_qubits_list.append(control_register[i])

    if rotation_type == 'ry':
        quantum_circuit.mcry(angle, ctrl_qubits_list, target_register[target_index])

    elif rotation_type == 'rz':
        quantum_circuit.mcrz(angle, ctrl_qubits_list, target_register[target_index])

    elif rotation_type == 'rx':
        quantum_circuit.mcrx(angle, ctrl_qubits_list, target_register[target_index])

    else:
        raise Exception("Unavailable rotation type")

    return quantum_circuit


def _register_step(feature, quantum_circuit, quantum_data, ancila, n_dset_size_qubits):
    """
        Auxiliary procedure that applies the multicontrolled rotations step
        in Park's method
    :param feature: Continuous value to be encoded in the phase;
    :param quantum_circuit: Quantum Circuit Object where the rotations are to be encoded
    :param quantum_data: Quantum Register object for the qubits dedicated to the data
    :param ancila: Quantum Register object for the ancila
    :param n_dset_size_qubits: Number of qubits necessary to encode the data in to the states
    :return:Quantum circuit updated with the register step
    """

    gamma = 0
    beta = 0

    if isinstance(feature, complex):
        phase = np.sqrt(np.power(np.absolute(feature), 2))
        gamma = 2 * np.arcsin(phase)
        beta = 2 * np.arcsin(feature.imag / phase)
    else:
        gamma = 2 * np.arcsin(feature)

    quantum_circuit = _apply_apply_multi_controlled_rotation(gamma, quantum_circuit,
                                                             quantum_data, ancila, 0,
                                                             rotation_type='ry')
    quantum_circuit = _apply_apply_multi_controlled_rotation(beta, quantum_circuit,
                                                             quantum_data, ancila, 0,
                                                             rotation_type='rz')

    return quantum_circuit


def park_quantum_circuit(features, n_feature_qubits, n_dset_size_qubits, with_barriers=False):
    """
        Generates the quantum circuit for the method of Park, et al.
        To encode complex feature vectors in to a quantum state
    :param features: List of tuples vector to be encoded in the quantum state,
                    Tuples need to be in the format (v, b).
                    Where v is the value to be encoded in the phase,
                    And b the binary string associated to it
    :param n_feature_qubits: Number of qubits needed for encoding the features in to the
                             quantum state
    :param n_dset_size_qubits: Number of qubits needed for encoding the index qubit
                               for each feature in the dataset
    :param with_barriers:Boolean, add the barriers in the quantum circuit for better visualisation
                        when printing the circuit
    :return: Quantum Circuit object generated to perform the method of Park's, et al.
    """

    # Quantum registers dedicated to the encoding procedure
    quantum_data = QuantumRegister(n_feature_qubits)
    ancila = QuantumRegister(1)
    quantum_index = QuantumRegister(n_dset_size_qubits)

    # Classical Register dedicated to post selection
    post_selection_reg = ClassicalRegister(1)

    circuit = QuantumCircuit(quantum_data, ancila, quantum_index, post_selection_reg)

    initialize_index = False

    if len(features) > 2:
        initialize_index = True

    circuit = _initialize_state(circuit, initialize_index=initialize_index)

    for feature, pattern in features:

        # FLIP
        circuit = _qubitwise_not(pattern, circuit, quantum_data)

        if with_barriers:
            circuit.barriers()# pylint: disable=maybe-no-member

        # REGISTER
        circuit = _register_step(feature, circuit, quantum_data, ancila, n_dset_size_qubits)

        if with_barriers:
            circuit.barriers()# pylint: disable=maybe-no-member

        # FLOP
        circuit = _qubitwise_not(pattern, circuit, quantum_data)

    # MEASURING ANCILA FOR POST SELECTION OF THE STATE
    circuit.measure(ancila[0], post_selection_reg[0])

    return circuit