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
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import transpile
from qiskit.circuit.library import RYGate, CZGate, MCMT
from qiskit.extensions import UnitaryGate, UCRYGate, UCRZGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.synthesis import two_qubit_decompose
from qiskit.quantum_info import Operator
from qclib.gates.ucr import ucr
from qclib.gates.uc_gate import UCGate
from qclib.decompose2q import TwoQubitDecomposeUpToDiagonal


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
    if decomposition == 'qsd' and size > 4:
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
        # TODO: Replace this recursion with a mathematical expression.
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


def _apply_a2(circ):
    # This code is part of Qiskit.
    #
    # (C) Copyright IBM 2017, 2019.
    #
    # This code is licensed under the Apache License, Version 2.0. You may
    # obtain a copy of this license in the LICENSE.txt file in the root directory
    # of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
    #
    # Any modifications or derivative works of this code must retain this
    # copyright notice, and modified files need to carry a notice indicating
    # that they have been altered from the originals.

    # from qiskit import transpile
    # from qiskit.quantum_info import Operator

    # from qiskit.extensions.unitary import UnitaryGate
    # import qiskit.extensions.unitary

    decomposer = TwoQubitDecomposeUpToDiagonal()
    ccirc = transpile(circ, basis_gates=["u", "cx", "qsd2q"], optimization_level=0)
    ind2q = []
    # collect 2q instrs
    for i, instr_context in enumerate(ccirc.data):
        instr, _, _ = instr_context
        if instr.name == "qsd2q":
            ind2q.append(i)
    # rolling over diagonals
    ind2 = None  # lint
    mat2 = None
    qargs = None
    cargs = None

    for ind1, ind2 in zip(ind2q[0:-1:], ind2q[1::]):
        # get neigboring 2q gates separated by controls
        instr1, qargs, cargs = ccirc.data[ind1]
        mat1 = Operator(instr1).data
        instr2, _, _ = ccirc.data[ind2]
        mat2 = Operator(instr2).data
        # rollover
        dmat, qc2cx = decomposer(mat1)
        ccirc.data[ind1] = (qc2cx.to_gate(), qargs, cargs)
        mat2 = mat2 @ dmat
        # ccirc.data[ind2] = (qiskit.extensions.unitary.UnitaryGate(mat2), qargs, cargs)
        ccirc.data[ind2] = (UnitaryGate(mat2), qargs, cargs)
    if mat2 is not None:
        qc3 = two_qubit_decompose.two_qubit_cnot_decompose(mat2)
        ccirc.data[ind2] = (qc3.to_gate(), qargs, cargs)
    return ccirc

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
    for col_idx in range(2**n_qubits):
        for row_idx in range(col_idx+1, 2**n_qubits):
            # create q_matrix
            Q = np.eye(2**n_qubits, dtype=gate.dtype)
            # computing norm
            norm = np.linalg.norm([gate[row_idx-1, col_idx], 
                                   gate[row_idx, col_idx]])

            # computing Q
            Q[row_idx-1, col_idx] = gate[row_idx-1, col_idx] / norm
            Q[row_idx, col_idx] = gate[row_idx, col_idx] / norm

            Q[row_idx-1, col_idx+1] = gate[row_idx, col_idx] / norm
            Q[row_idx, col_idx+1] = -gate[row_idx-1, col_idx] / norm

            # applyting Q to the unitary
            gate = Q @ gate
            gate_sequence.append(Q)
    return np.array(gate_sequence)

def _get_row_col(Q, dim_matrix):
    for row_idx in range(dim_matrix):
        for col_idx in range(row_idx):
            if Q[row_idx][col_idx] != 0:
                b = Q[row_idx][col_idx]
                a = Q[col_idx][col_idx]
                col = col_idx
                row = row_idx

    return a, b, row, col

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
        #Find the position and values of a,b
        a,b,row,col = _get_row_col(Q, 2**3)
        print(row, col)

        #bit-wise operation to determine the qubits
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
        print(row_qubits)
        print(col_qubits)
        print(diff_qubits)
        print(n_diff)

        #for m in range(n_qubits):
        #   base = 2**m
        #    if(base & diff):
        #        #diff_qubits.append(m)
        #        n_diff+=1
        while(n_diff>1):
            n_diff= n_diff-1
        U = [[a,b], [b, -a]]
        #print(U)
        print(diff_qubits)
        gate = UnitaryGate(U)
        qubits_list=[]
        for m in range(n_qubits):
            if(diff_qubits[m]==0):
                if(row_qubits[m]==0):
                    circuit.x(m)
                qubits_list.append(m)
            else:
                diffqubit = m
        qubits_list.append(diffqubit)
        print(qubits_list)
        circuit.append(MCMT(gate, n_qubits-1,1), qubits_list)
        circuit.draw()
                





    return circuit
    