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

"""
This module provides access to functions to
implement generic quantum computations.
"""

from math import ceil, log2
import numpy as np
import scipy as sp
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import RYGate, CZGate
from qiskit.circuit.library import UnitaryGate, UCRYGate, UCRZGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.synthesis.unitary.qsd import _apply_a2
from qiskit.circuit.library import UCGate
from qclib.gates.ucr import ucr



def unitary(gate, decomposition="qsd", iso=0, apply_a2=True):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    circuit = build_unitary(gate, decomposition, iso)
    if decomposition == "qsd" and apply_a2:
        return _apply_a2(circuit)

    return circuit


def build_unitary(gate, decomposition="qsd", iso=0):
    """
    Implements a generic quantum computation from a
    unitary matrix gate using the cosine sine decomposition.
    """
    size = len(gate)
    if decomposition != 'qr' and size > 4:
        n_qubits = int(log2(size))

        qubits = QuantumRegister(n_qubits)
        circuit = QuantumCircuit(qubits)

        right_gates, theta, left_gates = sp.linalg.cossin(
            gate, size / 2, size / 2, separate=True
        )

        # Left circuit
        if iso:
            gate_left = build_unitary(
                left_gates[0], decomposition=decomposition, iso=iso - 1
            )
            circuit = circuit.compose(gate_left, qubits[:-1])
        else:
            gate_left = _unitary(list(left_gates), n_qubits, decomposition)
            circuit.compose(gate_left, qubits, inplace=True)

        # Middle circuit
        # Last CZGate is ommited and absorved into the neighboring multiplexor.
        ucry = ucr(RYGate, list(2 * theta), CZGate, False)

        circuit.append(
            ucry.to_instruction(), [n_qubits - 1] + list(range(n_qubits - 1))
        )
        # Optimization (A.1) from "Synthesis of Quantum Logic Circuits".
        # Last CZGate from ucry is absorbed here.
        right_gates[1][:, len(theta) // 2 :] = -right_gates[1][:, len(theta) // 2 :]

        # Right circuit
        gate_right = _unitary(list(right_gates), n_qubits, decomposition)
        circuit.compose(gate_right, qubits, inplace=True)

        return circuit
    elif decomposition == 'qr':
        return _qrd(gate)

    circuit = QuantumCircuit(int(log2(size)), name="qsd2q")
    circuit.append(UnitaryGate(gate), circuit.qubits)

    return circuit


def _unitary(gate_list, n_qubits, decomposition="qsd"):

    if decomposition == "csd":
        if len(gate_list[0]) == 2:
            qubits = QuantumRegister(n_qubits)
            circuit = QuantumCircuit(qubits)
            circuit.append(UCGate(gate_list), qubits[[0]] + qubits[1:])
            return circuit

        return _csd(gate_list, n_qubits)

    # QSD
    return _qsd(*gate_list)


# CSD decomposition


def _csd(gate_list, n_qubits):
    left, mid, right = _multiplexed_csd(gate_list)

    gate_left = _unitary(left, n_qubits, decomposition="csd")
    gate_right = _unitary(right, n_qubits, decomposition="csd")

    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)

    circuit = circuit.compose(gate_left, qubits)

    target = int(n_qubits - log2(len(left)))
    control = list(range(0, target)) + list(range(target + 1, n_qubits))
    circuit.append(UCRYGate(list(mid)), [target] + control)

    circuit = circuit.compose(gate_right, qubits)

    return circuit


def _multiplexed_csd(gate_list):
    left = []
    mid = []
    right = []
    size = len(gate_list[0])
    for gate in gate_list:
        right_gates, theta, left_gates = sp.linalg.cossin(
            gate, size / 2, size / 2, separate=True
        )

        left = left + list(left_gates)
        right = right + list(right_gates)
        mid = mid + list(2 * theta)

    return left, mid, right


# QSD decomposition


def _qsd(gate1, gate2):
    n_qubits = int(log2(len(gate1))) + 1
    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)

    list_d, gate_v, gate_w = _compute_gates(gate1, gate2)

    # Left circuit
    left_gate = build_unitary(gate_w, "qsd")
    circuit.append(left_gate.to_instruction(), qubits[0:-1])

    # Middle circuit
    circuit.append(UCRZGate(list(-2 * np.angle(list_d))), qubits[[-1]] + qubits[0:-1])

    # Right circuit
    right_gate = build_unitary(gate_v, "qsd")
    circuit.append(right_gate.to_instruction(), qubits[0:-1])

    return circuit


def _closest_unitary(matrix):
    svd_u, _, svd_v = np.linalg.svd(matrix)
    return svd_u.dot(svd_v)


def _compute_gates(gate1, gate2):

    d_square, gate_v = np.linalg.eig(gate1 @ gate2.conj().T)
    list_d = np.sqrt(d_square, dtype=complex)
    gate_d = np.diag(list_d)

    if not is_unitary_matrix(gate_v):
        # degeneracy
        gate_v = _closest_unitary(gate_v)

    gate_w = gate_d @ gate_v.conj().T @ gate2

    return list_d, gate_v, gate_w


def cnot_count(gate, decomposition="qsd", method="estimate", iso=0, apply_a2=True):
    """
    Count the number of CNOTs to decompose the unitary.
    """
    if method == "estimate":
        return _cnot_count_estimate(gate, decomposition, iso, apply_a2)

    # Exact count
    circuit = unitary(gate, decomposition, iso, apply_a2)
    transpiled_circuit = transpile(
        circuit, basis_gates=["u", "cx"], optimization_level=0
    )
    count_ops = transpiled_circuit.count_ops()
    if "cx" in count_ops:
        return count_ops["cx"]

    return 0


def _cnot_count_estimate(gate, decomposition="qsd", iso=0, apply_a2=True):
    """
    Estimate the number of CNOTs to decompose the unitary.
    """
    n_qubits = int(log2(gate.shape[0]))

    if n_qubits == 1:
        return 0

    if n_qubits == 2:
        return 3

    if decomposition == "csd":
        # Table 1 from "Synthesis of Quantum Logic Circuits", Shende et al.
        return int(ceil(4**n_qubits - 2 * 2**n_qubits)) - 1

    if iso:
        # Replace this recursion with a mathematical expression.
        last_2q_gate_cnot = 1 if apply_a2 else 0
        return _cnot_count_iso(n_qubits, iso, apply_a2) + last_2q_gate_cnot

    # Upper-bound expression for the unitary decomposition QSD
    # Table 1 from "Synthesis of Quantum Logic Circuits", Shende et al.
    if apply_a2:
        # l=2 with optimizations (A.1) and (A.2) from "Synthesis of Quantum Logic Circuits".
        return int(
            ceil((23 / 48) * 2 ** (2 * n_qubits) - 3 / 2 * 2**n_qubits + 4 / 3)
        )

    # l=2 only with optimization (A.1).
    # Optimization (A.2) counts 4**(n_qubits-2)-1 CNOT gates.
    return (
        4 ** (n_qubits - 2)
        - 1
        + int(ceil((23 / 48) * 2 ** (2 * n_qubits) - 3 / 2 * 2**n_qubits + 4 / 3))
    )


def _cnot_count_iso(n_qubits, iso, apply_a2=True):
    if n_qubits > 2:
        # Left circuit
        if iso:
            iso_cnot = 1 if n_qubits - 1 == 2 else 0
            gate_left = _cnot_count_iso(n_qubits - 1, iso - 1, apply_a2) + iso_cnot
        else:
            gate_left = _cnot_count_iso_qsd(n_qubits, apply_a2)

        # Middle circuit
        ucry = 2 ** (n_qubits - 1) - 1

        # Right circuit
        gate_right = _cnot_count_iso_qsd(n_qubits, apply_a2)

        return gate_left + ucry + gate_right

    if apply_a2:
        return 2

    return 3


def _cnot_count_iso_qsd(n_qubits, apply_a2):
    # Left circuit
    left_gate = _cnot_count_iso(n_qubits - 1, 0, apply_a2)

    # Middle circuit
    middle_gate = 2 ** (n_qubits - 1)

    # Right circuit
    right_gate = _cnot_count_iso(n_qubits - 1, 0, apply_a2)

    return left_gate + middle_gate + right_gate

# QR decomposition

def _qrd(gate: np.ndarray):
    """"""

    n_qubits = int(np.log2(len(gate)))

    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)

    gate_sequence = _build_qr_gate_sequence(gate, n_qubits)
    circuit = _build_qr_circuit(gate_sequence, n_qubits)

    return circuit

def _build_qr_gate_sequence(gate, n_qubits):
    gate_sequence = []
    for col_idx in range(2**n_qubits - 1):
        for row_idx in range(col_idx+1, 2**n_qubits):
            # create q_matrix
            
            Q = np.eye(2**n_qubits, dtype=gate.dtype)
            # computing norm
            norm = np.linalg.norm([gate[col_idx, col_idx], 
                                   gate[row_idx, col_idx]])

            # computing Q
            a = gate[col_idx, col_idx] / norm
            b = gate[row_idx, col_idx] / norm
            Q[col_idx, col_idx] = np.conj(a)
            Q[col_idx, row_idx] = np.conj(b)

            Q[row_idx, col_idx] = b
            Q[row_idx, row_idx] = -a

            # applyting Q to the unitary
            gate = Q @ gate
            gate_sequence.append(Q.conj().T)

    gate_sequence.append(gate)
    gate_sequence = list(reversed(gate_sequence))
    return np.array(gate_sequence)

def _get_row_col(Q, dim_matrix):
    a, b, c, d = 1., 0., 0., 1.
    for row_idx in range(dim_matrix):
        for col_idx in range(row_idx):
            if Q[row_idx][col_idx] != 0 and np.not_equal(Q[row_idx][col_idx], 1):
                
                a = Q[col_idx][col_idx]
                b = Q[row_idx][col_idx]
                c = Q[col_idx][row_idx]
                d = Q[row_idx][row_idx]
                col = col_idx
                row = row_idx
    return a, b, c, d, row, col

def _apply_MCXs(n_qubits, row_qubits_new, col_qubits_new):
    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)
    for m in range(n_qubits):
        if(row_qubits_new[m]==0 and  col_qubits_new[m]==1):
            qubits_list=[]
            memory = np.ones(shape= n_qubits, dtype = int)
            for n in range(n_qubits):
                if(n!=m):
                    if(row_qubits_new[n]==0):
                        circuit.x(n)
                        memory[n] = 0
                    qubits_list.append(n)
            qubits_list.append(m)
            #for the target qubit
            memory[m] = 2
            circuit.append(MCXGate(n_qubits-1), qubits_list)
            row_qubits_new[m]=1
            for n in range(n_qubits):
                if(n!=m and row_qubits_new[n]==0):
                        circuit.x(n)
            break
        if(row_qubits_new[m]==1 and  col_qubits_new[m]==0):
            qubits_list=[]
            memory = np.ones(shape= n_qubits, dtype = int)

            for n in range(n_qubits):
                if(n!=m):
                    if(col_qubits_new[n]==0):
                        circuit.x(n)
                        memory[n] = 0
                    qubits_list.append(n)
            qubits_list.append(m)
            #for the target qubit
            memory[m] = 2
            circuit.append(MCXGate(n_qubits-1), qubits_list)
            col_qubits_new[m]=1
            for n in range(n_qubits):
                if(n!=m and col_qubits_new[n]==0):
                        circuit.x(n)
            
            break
    return circuit, memory, row_qubits_new, col_qubits_new

def _undo_MCXs(prep_gates, n_qubits ):
    sz = len(prep_gates)
    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)
    for i in range(sz):
        qubits_list=[]
        #We are going through the list backwards
        aux = prep_gates[sz-1-i]
        for m in range(n_qubits):
            if(aux[m]==0):
                qubits_list.append(m)
                circuit.x(m)
            if(aux[m]==1):
                qubits_list.append(m)
            if(aux[m]==2):
                target = m
        qubits_list.append(target)    
        circuit.append(MCXGate(n_qubits-1), qubits_list)
        #Don't forget the X/NOT at the end
        for m in range(n_qubits):
            if(aux[m]==0):
                circuit.x(m)
    return circuit

''''''
def _build_qr_circuit(gate_sequence, n_qubits):
    '''
    This function was coded for real numbers only
    Input:
    gate_sequence: Sequence of unitary matrixes with the majority of the elements being 1s in the diagonal, except for 4 elements:
    "a and -a" in the diagonal and "2 elements b" outside the diagonal. Since they are unitary, the position of the two b's outside the diagonal are related
    n_qubits: Number of qubits

    '''
    #inverter e tirar o complex conjugado de gate_sequence
    dim_matrix = 2**n_qubits
    qubits = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(qubits)
    for Q in gate_sequence:

        a, b, c, d, row, col = _get_row_col(Q, 2**n_qubits)

        col_qubits = []
        row_qubits = []
        diff_qubits = np.zeros(n_qubits)
        n_diff = 0
        for m in range(n_qubits):
            base = 2**m
            if(col & base == 0):
                col_qubits.append(0)
            else:
                col_qubits.append(1)
            if (row & base == 0):
                row_qubits.append(0)
            else:
                row_qubits.append(1)
            if(row_qubits[m] != col_qubits[m]):
                n_diff +=1
                diff_qubits[m]=1
        row_qubits_new = row_qubits
        col_qubits_new = col_qubits
        # Apply MCXs to only have one qubit different
        #prep_gates and memory saves information about the gates we used, so we can use them again later
        prep_gates =[]
        
        while(n_diff>1):
            diff_circ, memory, row_qubits_new, col_qubits_new = _apply_MCXs(n_qubits, row_qubits_new, col_qubits_new)
            qubits_list = range(n_qubits)
            circuit.append(diff_circ, qubits_list)
            prep_gates.append(memory)
            n_diff-=1     
        U = np.array([[a, c], 
                      [b, d]])
        gate = UnitaryGate(U)
        qubits_list=[]
        for m in range(n_qubits):
            if(col_qubits_new[m]==row_qubits_new[m]):
                if(row_qubits_new[m]==0):
                    circuit.x(m)
                qubits_list.append(m)
            else:
                diffqubit = m
        qubits_list.append(diffqubit)
        circuit.append(MCMT(gate, n_qubits-1,1), qubits_list)
        for m in range(n_qubits):
            if(col_qubits_new[m]==row_qubits_new[m]):
                if(row_qubits[m]==0):
                    circuit.x(m)
        #Do all the MCXs again
        diff_circ = _undo_MCXs(prep_gates, n_qubits )
        qubits_list = range(n_qubits)
        circuit.append(diff_circ, qubits_list)
        

    return circuit
