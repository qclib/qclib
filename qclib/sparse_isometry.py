"""
Malvetti, Emanuel, Raban Iten, and Roger Colbeck.
"Quantum circuits for sparse isometries." Quantum 5 (2021): 412.
TODO: fix number of cx gates
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qclib.state_preparation.pivot import PivotInitialize



def build_state_dict(state):
    """
    Builds a dict of the non zero amplitudes with their
    associated binary strings as follows:
      { '000': <value>, ... , '111': <value> }
    Args:
      state: The classical description of the state vector
    """
    n_qubits = np.ceil(np.log(len(state))).astype(int)
    state_dict = {}
    for value_idx, value in enumerate(state):
        if value != 0:
            binary_string = f"{value_idx:0{n_qubits}b}"
            state_dict[binary_string] = value
    return state_dict


def householder_reflection_zero(num_qubits, phi=np.pi):
    """
    Perform a Houlseholder reflection H_0^ phi
    (https://arxiv.org/pdf/2006.00016.pdf section 3 )

    num_qubits: number of qubits
    phi: phase phi with respect to the vector v
    """

    qc = QuantumCircuit(1)
    qc.unitary([[np.e ** (1j * phi), 0], [0, 1]], [0])
    qc_ctrl = qc.control(num_qubits - 1, ctrl_state=(num_qubits - 1) * "0")

    return qc_ctrl


def generalized_householder_reflection(data_list, x, y):
    """
    Perform a Householder reflection H_v^ phi
    (https://arxiv.org/pdf/2006.00016.pdf section 3 )
    (https://ieeexplore.ieee.org/document/622959 section 1)

    data_list: desired vector v (type = list)
    x: input vector
    y: desired vector
    """

    # Angle involved in Householder reflection
    a = x
    b = y
    z = a - b
    y = (z.conj().T @ b) / (z.conj().T @ a)
    phi = np.angle(y)

    # Number of qubits
    num_qubits = int(np.log2(len(data_list)))

    # Converts the list into a dictionary for use in PivotInitialize
    data = build_state_dict(data_list)

    # Initializing the quantum circuit of the size of the received vector
    circuit = QuantumCircuit(num_qubits)

    # Initializes the state preparation SP
    sp = PivotInitialize(data).definition

    # Assembly of H_v^\phi
    circuit.compose(sp.inverse(), inplace=True)

    h = householder_reflection_zero(num_qubits, phi)
    circuit = circuit.compose(h, list(np.flip(np.arange(num_qubits))))

    circuit.compose(sp, inplace=True)

    return circuit


def householder_decomposition(isometry):
    """
    Perform a Householder Decomposition
    (https://arxiv.org/pdf/2006.00016.pdf section 4 )

    isometry: desired isometry
    """

    # We select the number of rows and columns for the chosen isometry
    num_isometry_rows = len(isometry[:, 0])
    num_isometry_columns = len(isometry[0, :])

    # We create a diagonal matrix of the same dimensions as the isometry
    diagonal_matrix = np.array(
        [
            [1 if ii == jj else 0 for jj in range(num_isometry_columns)]
            for ii in range(num_isometry_rows)
        ]
    ).astype(complex)

    # We create a circuit to store all the h_reflection Householder operators
    circuit = QuantumCircuit(np.log2(num_isometry_rows))

    ii = 0
    while not np.allclose(isometry, diagonal_matrix):
        x = isometry[:, ii]
        y = diagonal_matrix[:, ii]
        v = y - x
        v = v / np.linalg.norm(v)
        v = list(v)
        h_reflection = generalized_householder_reflection(v, x, y)
        circuit.compose(h_reflection, inplace=True)
        isometry = Operator(h_reflection).data @ isometry

        if ii >= (num_isometry_columns - 1):
            break

        ii += 1

    return circuit
