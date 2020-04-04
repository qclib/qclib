"""
    This module is dedicated to the implementation of procedures that
    are not directly related to the implementation of quantum circuits
    but can be used as a tool to preprocess or compute necessary data
    or elements so that the circuits can be properly executed.
"""
import numpy as np


def transform_dataset(dataset):
    """
        Auxiliary procedure for the Park's and Adapted Tugenberger's method.
        Transforms the dataset in to a list in which each entry is a list
        of tuples, such as :

        [[(x_0, b_0), (x_1, b_1), ..., (x_n, b_n)],
            ...
          ,[(x_0, b_0), (x_1, b_1), ... , (x_n, b_n)]]

        Where x_y is the real/complex value in the y-th component of
        feature vector and b_y is the binary string associated to the
        y-th value.
        This facilitates the storage of sparse vectors into the state
    :param dataset: Array or list of values to be transformed
    :return: List of tuples and the number of qubits necessary to
            encode the features into the state
    """

    if not isinstance(dataset, (list, np.ndarray)):
        raise Exception("Expected List of Numpy " +
                        "ndarray types, but got {} instead".format(type(dataset)))

    temp_dataset = dataset
    # Verifying if the dataset is a unidimesional iterable
    if not isinstance(dataset[0], (list, np.ndarray)):

        if isinstance(dataset, list):
            temp_dataset = [temp_dataset]
        elif isinstance(dataset, np.ndarray):
            temp_dataset = np.array([dataset.tolist()])

    # Number of qubits for encoding the feature into the state
    n_qbits = int(np.ceil(np.log2(len(temp_dataset[0]))))

    transfomed_dataset = []

    for fv_index, feature_vector in enumerate(temp_dataset):

        transfomed_dataset.append([])

        for ft_index, feature in enumerate(feature_vector):

            if feature != 0:
                binary_state = format(ft_index, 'b').zfill(n_qbits)
                transfomed_dataset[fv_index].append((feature, binary_state))

    return transfomed_dataset, n_qbits


def build_list_of_quibit_objects(quantum_register):
    """
        Buid a list of Qubit objects to be used as
        input to some procedure of the qiskit framework
    :param quantum_register: Quantum register with the qubits
    :return: Qubits list
    """
    qubits_list = []

    for i in range(quantum_register.size):
        qubits_list.append(quantum_register[i])

    return qubits_list
