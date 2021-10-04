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

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qclib.state_preparation.mottonen import initialize as mottonen
                        
# pylint: disable=maybe-no-member

def initialize(state, rank=0):
    """ State preparation using Schmidt decomposition arXiv:1003.5760

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
    
    s = np.copy(state)

    size = len(s)
    n_qubits = int(np.log2(size))

    r = n_qubits % 2

    n = int(2**(n_qubits//2))
    m = int(2**(n_qubits//2 + r))
    s.shape = (n, m)

    u, d, v = np.linalg.svd(s)
    
    if (rank >= 1 and rank <= n):
        # This code is equivalent to the commented one below. It is more compact, but the one below is easier to read.
        d = d[:rank]
        u = u[:,:rank]
        v = v[:rank ,:]
        if (not np.log2(rank).is_integer()): # To use isometries, the rank needs to be a power of 2.
            s = u @ np.diag(d) @ v 
            """
            s = np.zeros((n, m), dtype=complex)
            for i in range(rank):
                s += d[i] * np.outer(u.T[i], v[i])
            """ 
            u, d, v = np.linalg.svd(s)

        d = np.concatenate((d, [0]*(s.shape[0]-d.shape[0])))
    else:
        rank = n

    v = v[:n, :] # Isometry n to m.

    d = d / np.linalg.norm(d)

    A = QuantumRegister(n_qubits//2 + r)
    B = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(A, B)
    
    if len(d) > 2:
        circ = initialize(d)
    else:
        circ = mottonen(d)
    circuit.compose(circ, B, inplace=True)
        
    for k in range(int( np.ceil(np.log2(rank)) )):
        circuit.cx(B[k], A[k])

    def encode(U, reg):                           # Encodes the data using the most appropriate method.
        if (U.shape[1] == 1):                     # State preparation.
            gate_u = initialize(U[:,0])
        elif (U.shape[0] > U.shape[1]):           # Isometry decomposition.
            from qclib.isometry import decompose
            gate_u = decompose(U, scheme='knill') 
        else:                                     # Unitary.
            from qclib.unitary import unitary
            gate_u = unitary(U, 'qsd')
        
        circuit.compose(gate_u, reg, inplace=True) # Apply gate U to the register.

    encode(u, B)
    encode(v.T, A)
    
    return circuit




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
    
    s = np.copy(state)

    size = len(s)
    n_qubits = int(np.log2(size))

    r = n_qubits % 2

    s.shape = (int(2**(n_qubits//2)), int(2**(n_qubits//2 + r)))

    u, d, v = np.linalg.svd(s)
        
    d = d / np.linalg.norm(d)

    A = QuantumRegister(n_qubits//2 + r)
    B = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(A, B)

    if len(d) > 2:
        circ = initialize_original(d)
    else:
        circ = mottonen(d)
    circuit.compose(circ, B, inplace=True)

    for k in range(int(n_qubits//2)):
        circuit.cx(B[k], A[k])

    from qclib.unitary import unitary
    
    gate_u = unitary(u, 'qsd')
    gate_v = unitary(v.T, 'qsd')

    circuit.compose(gate_u, B, inplace=True) # apply gate U to the first register
    circuit.compose(gate_v, A, inplace=True) # apply gate V to the second register

    return circuit
