# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" State preparation with Schmidt decomposition """

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.state_preparation.mottonen import initialize as mottonen
from qclib.unitary import unitary
from qclib.isometry import decompose

# pylint: disable=maybe-no-member

def initialize(state, rank=0):
    """ State preparation using Schmidt decomposition

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize(state)

    Parameters
    ----------
    state: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    rank: int
        ``state`` low-rank approximation (1 <= ``rank`` < 2**(n_qubits//2)).
        If ``rank`` is not in the valid range, it will be ignored and the full
        ``rank`` will be used.

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """

    state_copy = np.copy(state)

    hsa, hsb, n_a, n_b = _init_params(state_copy)

    state_copy.shape = (n_a, n_b)

    u_matrix, diag, v_matrix = np.linalg.svd(state_copy)

    if 1 <= rank <= n_a:
        # This code is equivalent to the commented one below.
        # It is more compact, but the one below is easier to read.
        diag = diag[:rank]
        u_matrix = u_matrix[:, :rank]
        v_matrix = v_matrix[:rank, :]
        if not np.log2(rank).is_integer(): # To use isometries, the rank needs to be a power of 2.
            state_copy = u_matrix @ np.diag(diag) @ v_matrix

            # s = np.zeros((n, m), dtype=complex)
            # for i in range(rank):
            #     s += d[i] * np.outer(u.T[i], v[i])

            u_matrix, diag, v_matrix = np.linalg.svd(state_copy)

        diag = np.concatenate((diag, [0]*(state_copy.shape[0]-diag.shape[0])))
    else:
        rank = n_a

    v_matrix = v_matrix[:n_a, :] # Isometry n to m.

    diag = diag / np.linalg.norm(diag)

    circuit = QuantumCircuit(hsa, hsb)

    if len(diag) > 2:
        circ = initialize(diag)
    else:
        circ = mottonen(diag)
    circuit.compose(circ, hsb, inplace=True)

    for k in range(int( np.ceil(np.log2(rank)) )):
        circuit.cx(hsb[k], hsa[k])

    def encode(matrix, reg):
        """
        Encodes the data using the most appropriate method.
        """
        # State preparation
        if matrix.shape[1] == 1:
            gate_u = initialize(matrix[:, 0])

        # Isometry decomposition.
        elif matrix.shape[0] > matrix.shape[1]:
            gate_u = decompose(matrix, scheme='knill')
        else:
            gate_u = unitary(matrix, 'qsd')

        circuit.compose(gate_u, reg, inplace=True) # Apply gate U to the register.

    encode(u_matrix, hsb)
    encode(v_matrix.T, hsa)

    return circuit


def _init_params(state_copy):
    size = len(state_copy)
    n_qubits = int(np.log2(size))
    is_odd = n_qubits % 2
    n_a = int(2 ** (n_qubits // 2))
    n_b = int(2 ** (n_qubits // 2 + is_odd))
    hsa = QuantumRegister(n_qubits // 2 + is_odd)
    hsb = QuantumRegister(n_qubits // 2)
    return hsa, hsb, n_a, n_b


#import deprecation
#@deprecation.deprecated(deprecated_in="0.0.7",
#                        details="Use the initialize function instead")
def initialize_original(state):
    """ State preparation using Schmidt decomposition arXiv:1003.5760.
        This function implements the original algorithm as defined in arXiv:1003.5760.
        It is kept here for didactic reasons.
        The ``initialize`` function should preferably be used.
        This function is used in isometry.py.

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize_original(state)

    Parameters
    ----------
    state: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """

    state_copy = np.copy(state)

    size = len(state_copy)
    n_qubits = int(np.log2(size))

    is_odd = n_qubits % 2

    state_copy.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + is_odd)))

    u_matrix, diag, v_matrix = np.linalg.svd(state_copy)

    diag = diag / np.linalg.norm(diag)

    hsa = QuantumRegister(n_qubits//2 + is_odd)
    hsb = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(hsa, hsb)

    if len(diag) > 2:
        circ = initialize_original(diag)
    else:
        circ = mottonen(diag)
    circuit.compose(circ, hsb, inplace=True)

    for k in range(int(n_qubits//2)):
        circuit.cx(hsb[k], hsa[k])

    gate_u = unitary(u_matrix, 'qsd')
    gate_v = unitary(v_matrix.T, 'qsd')

    circuit.compose(gate_u, hsb, inplace=True) # apply gate U to the first register
    circuit.compose(gate_v, hsa, inplace=True) # apply gate V to the second register

    return circuit
