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
from qclib.unitary import unitary
from qclib.isometry import decompose

# pylint: disable=maybe-no-member

def initialize(state, low_rank=0):
    """ State preparation using Schmidt decomposition arXiv:1003.5760

    For instance, to initialize the state a|0> + b|1>
        $ state = [a, b]
        $ circuit = initialize(state)

    Parameters
    ----------
    state: list of int
        A unit vector representing a quantum state.
        Values are amplitudes.

    low_rank: int
        ``state`` low-rank approximation (1 <= ``low_rank`` < 2**(n_qubits//2)).
        If ``low_rank`` is not in the valid range, it will be ignored and the effective
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
    
    
    rank = sum(j > 10**-16 for j in d)     # Effective rank ( rank \in {1, 2, ..., n} ).
    if (low_rank > 0 and low_rank < rank):
        rank = low_rank                    # Low-rank approximation
    
    k = int(2**np.ceil(np.log2(rank)))     # To use isometries, the rank needs to be a power of 2.
    
    u = u[:,:k]                            # Matrix u can be a unitary (k=n) or isometry (k<n).
    v = v[:k,:]                            # If n<m, v.T is always an isometry of log2(k) to log2(m) (k<=n).
                                           # If n=m, v.T can be a unitary (k=n) or isometry (k<n).
    d = d[:k]                              # The length of the state vector needs to be a power of 2.
    d[rank:] = np.zeros(k-rank)            # If k>rank, zeroes out the additional elements.
    
    if (rank == 1):
        d = np.concatenate((d, [0]))       # The length of the state vector needs to be a power of 2 and >1.
    

    d = d / np.linalg.norm(d)
    
    A = QuantumRegister(n_qubits//2 + r)
    B = QuantumRegister(n_qubits//2)

    circuit = QuantumCircuit(A, B)
    
    if len(d) > 2:
        circ = initialize(d)
    else:
        circ = mottonen(d)
        
    circuit.compose(circ, B[:int( np.log2( len(d) ) )], inplace=True)

    for j in range(int( np.log2( k ) )):           # Entangles only the necessary qubits, according to rank.
        circuit.cx(B[j], A[j])

    def encode(U, reg):                            # Encodes the data using the most appropriate method:
        if (U.shape[1] == 1):                      #   State preparation.
            if (U.shape[0] > 2):
                gate_u = initialize(U[:,0])
            else:
                gate_u = mottonen(U[:,0])
        elif (U.shape[0] > U.shape[1]):            #   Isometry decomposition.
            gate_u = decompose(U, scheme='knill') 
        else:                                      #   Unitary decomposition.
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
    
    gate_u = unitary(u, 'qsd')
    gate_v = unitary(v.T, 'qsd')

    circuit.compose(gate_u, B, inplace=True) # apply gate U to the first register
    circuit.compose(gate_v, A, inplace=True) # apply gate V to the second register

    return circuit
