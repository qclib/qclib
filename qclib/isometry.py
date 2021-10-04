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
import scipy
from qiskit import QuantumCircuit, QuantumRegister
from qclib.unitary import unitary
from qclib.state_preparation.bdsp import initialize

def decompose(V, scheme='knill'):
    v = np.copy(V)
    # change a row vector to a column vector (in the case of state preparation).
    if len(v.shape) == 1:
        v = v.reshape(v.shape[0], 1)

    N = v.shape[0]
    M = v.shape[1]
    n = np.log2(N)
    m = np.log2(M)
    
    _check_isometry(v, n, m)

    n = int(n)
    m = int(m)
    
    if (N == M): # The isometry v is already unitary.
        U = v
    else:        # The isometry v is extended to a unitary maximizing the numbers of eigenvalues with complex argument equal to zero.
        
        # The complex conjugate of the null space is the transformation that generates the state |v> from |0>.
        # V=UI_{2^n,2^m} => U^-1 V = I_{2^n,2^m}
        w = np.conj(scipy.linalg.null_space(v.T)) # The transposition was removed because it would be nullified in U[:, M:] = w.T .
        U = np.zeros((N,N), dtype=complex)
        U[:, :M] = v
        U[:, M:] = w                              # The transposition was removed because it nullified the transposition of the conjugate complex above (which was removed).
    
    if (scheme == 'csd'):
        pass
    elif (scheme == 'cbc'):
        pass

    return _knill(U, n)    




def _knill(U, n):
    if n < 2:
        raise ValueError(
            "Knill decomposition does not work on a 1 qubit isometry (N=2)."
        )

    from qclib.state_preparation.schmidt import initialize_original as schmidt

    A = QuantumRegister(n)
    circuit = QuantumCircuit(A)
        
    eigval, eigvec = np.linalg.eig(U)
    arg = np.angle(eigval)
        
    for i in range(2**n): # The eigenvalues are not necessarily ordered.
        if (np.abs(arg[i]) > 10**-15):
            state = eigvec[:,i]
            
            circuit.compose( schmidt(state).inverse(), A, inplace=True )

            circuit.x(list(range(n)))
            circuit.mcp(arg[i], list(range(n-1)), n-1)
            circuit.x(list(range(n)))

            circuit.compose( schmidt(state)          , A, inplace=True )
    
    return circuit







def _check_isometry(V, n, m):
    if not n.is_integer() or n < 0:
        raise ValueError(
            "The number of rows of the isometry is not a non negative power of 2."
        )
    if not m.is_integer() or m < 0:
        raise ValueError(
            "The number of columns of the isometry is not a non negative power of 2."
        )
    if m > n:
        raise ValueError(
            "The input matrix has more columns than rows and hence it can't be an isometry."
        )
    if not is_isometry(V, m):
        raise ValueError(
            "The input matrix has non orthonormal columns and hence it is not an isometry."
        )

def is_isometry(V, m):
    I = np.conj(V.T).dot(V)
    return np.allclose(I, np.eye(int(2**m)))

