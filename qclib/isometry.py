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
Implements the decomposition of isometries using the methods
defined at https://arxiv.org/abs/1501.06911.
"""

from math import log2
import numpy as np
import scipy
import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import DiagonalGate
from qiskit.circuit.library import UnitaryGate
from qiskit.circuit.library import UCGate
from qclib.unitary import unitary as decompose_unitary, cnot_count as unitary_cnot_count


def decompose(isometry: np.ndarray, scheme="ccd"):
    """
    Decompose an isometry from m to n qubits.
    In particular, it decomposes unitaries on n qubits (m=n) or prepare a
    quantum state on n qubits (m=0).
    `https://arxiv.org/abs/1501.06911`_.
    Args:
        isometry (list): an isometry from m to n qubits (n>=2 and m<=n), i.e., a
            complex 2^n x 2^m array with orthonormal columns.
        scheme (str): method to decompose the isometry ('knill', 'ccd', 'csd').
            Default is scheme='ccd'.
    Returns:
        QuantumCircuit: a quantum circuit with the isometry attached.
    Raises:
        ValueError: if isometry is not valid.
    """

    iso = isometry.astype(complex)
    if len(iso.shape) == 1:
        # change a row vector to a column vector (in the case of state preparation)
        iso = iso.reshape(iso.shape[0], 1)

    lines = iso.shape[0]
    cols = iso.shape[1]
    log_lines = log2(lines)
    log_cols = log2(cols)

    _check_isometry(iso, log_lines, log_cols)

    log_lines = int(log_lines)
    log_cols = int(log_cols)

    if scheme == "csd":
        return _csd(iso, log_lines, log_cols)

    if scheme == "ccd":
        return _ccd(iso, log_lines, log_cols)

    return _knill(iso, log_lines, log_cols)


# General
def _check_isometry(iso, log_lines, log_cols):
    if not log_lines.is_integer() or log_lines < 0:
        raise ValueError(
            "The number of rows of the isometry is not a non negative power of 2."
        )
    if not log_cols.is_integer() or log_cols < 0:
        raise ValueError(
            "The number of columns of the isometry is not a non negative power of 2."
        )
    if log_cols > log_lines:
        raise ValueError("The input matrix has more columns than rows.")
    if not _is_isometry(iso, log_cols):
        raise ValueError("The input matrix has non orthonormal columns.")


def _is_isometry(iso, log_cols):
    identity = np.conj(iso.T).dot(iso)
    return np.allclose(identity, np.eye(int(2**log_cols)))


# Cosine-Sine
def _csd(iso, log_lines, log_cols):
    unitary_gate = _extend_to_unitary(iso, log_lines, log_cols)

    return decompose_unitary(
        unitary_gate, decomposition="qsd", iso=log_lines - log_cols, apply_a2=True
    )


#   Knill
def _knill(iso, log_lines, log_cols):
    if log_lines < 2:
        raise ValueError(
            "Knill decomposition does not work on a 1 qubit isometry (N=2)."
        )

    unitary = _extend_to_unitary(iso, log_lines, log_cols)

    eigval, eigvec = np.linalg.eig(unitary)
    arg = np.angle(eigval)

    reg = QuantumRegister(log_lines)
    circuit = QuantumCircuit(reg)

    # pylint: disable=import-outside-toplevel
    from qclib.state_preparation import LowRankInitialize

    for i in range(2**log_lines):
        # The eigenvalues are not necessarily ordered.
        if np.abs(arg[i]) > 10**-7:
            state = eigvec[:, i]
            gate = LowRankInitialize(state)

            circuit.compose(gate.inverse(), reg, inplace=True)

            circuit.x(list(range(log_lines)))
            circuit.mcp(arg[i], list(range(log_lines - 1)), log_lines - 1)
            circuit.x(list(range(log_lines)))

            circuit.compose(gate, reg, inplace=True)

    return circuit


def _extend_to_unitary(iso, log_lines, log_cols):
    if log_lines == log_cols:
        # The isometry v is already unitary.
        unitary = iso
    else:
        # The isometry v is extended to a unitary maximizing
        # the numbers of eigenvalues with complex argument equal
        # to zero.
        null_space = np.conj(scipy.linalg.null_space(iso.T))
        # The complex conjugate of the null space is the transformation that
        # generates the state |v> from |0>. V=UI_{2^n,2^m} => U^-1 V=I_{2^n,2^m}
        # The transposition was removed because it would be nullified in
        # U[:, M:] = W.T .

        unitary = np.zeros((2**log_lines, 2**log_lines), dtype=complex)
        unitary[:, : 2**log_cols] = iso
        unitary[:, 2**log_cols :] = null_space
        # The transposition was removed because it nullified the
        # transposition of the conjugate complex above (which was removed).

    return unitary


# Column-by-column
def _ccd(iso, log_lines, log_cols):
    reg = QuantumRegister(log_lines)
    circuit = QuantumCircuit(reg)

    for k in range(2**log_cols):  # iteration through columns.
        g_k = _g_k(iso, log_lines, k)
        g_k.name = "G" + str(k)

        circuit.append(g_k.to_instruction(), reg)

    if log_cols > 0:
        # It clears the phases, as explained in the last sentence of the
        # first column on page 17.
        phases = np.angle(np.diagonal(iso[: 2**log_cols, : 2**log_cols]))
        diag = np.exp(-1j * phases)
        diag_gate = DiagonalGate(diag.tolist())
        circuit.append(diag_gate, list(range(log_cols)))

    return circuit.inverse()


def _g_k(iso, log_lines, col_index):
    # Gate G columns index k, to be created. Binary representation of column index k.
    # G_k's subgate bit index i (s in the paper).
    g_k = QuantumCircuit(log_lines)
    k_bin = f"{col_index:0{log_lines}b}"
    for i in range(log_lines):
        target = log_lines - i - 1
        control = list(range(target))
        ancilla = list(range(target + 1, log_lines))

        if _k_s(col_index, i) == 0 and _b(col_index, i + 1) != 0:
            # Condition defined in the first paragraph of the
            # second column on page 16. Generates single-qubit gate
            # matrix for the multicontrolled operation.
            unitary = _mc_unitary(iso, col_index, i)

            mcg = _mc_gate(unitary, log_lines, control + ancilla, target, k_bin)
            mcg = mcg.reverse_bits()  # Qiskit little-endian.
            _update_isometry(iso, mcg)

            # Append "mcg" to the operator "G_k".
            g_k.compose(mcg, list(range(log_lines)), inplace=True)

        # Generates single-qubit gates matrices for the uniformly controlled operation.
        unitaries = _uc_unitaries(iso, log_lines, col_index, i)
        ucg = _uc_gate(unitaries, log_lines, control, target)
        ucg = ucg.reverse_bits()  # Qiskit little-endian.
        _update_isometry(iso, ucg)

        # Append "ucg" to the operator "G_k".
        g_k.compose(ucg, list(range(log_lines)), inplace=True)
    return g_k


def _mc_gate(unitary, n_qubits, control, target, k_bin):
    gate = QuantumCircuit(n_qubits)

    controls = []
    for i in control:
        if k_bin[i] == "1":
            controls.append(i)

    unitaries = [np.identity(2) for _ in range(2 ** len(controls))]
    unitaries[-1] = unitary
    ucg = UCGate(unitaries, up_to_diagonal=True)
    gate.append(ucg, [target] + controls[::-1])

    return gate


def _uc_gate(unitaries, n_qubits, control, target):
    gate = QuantumCircuit(n_qubits)

    if len(control) > 0:
        # "control" is reversed due to UCGate implementation.
        uc_gate = UCGate(unitaries, up_to_diagonal=True)
        gate.append(uc_gate, [target] + control[::-1])
    else:
        # UCGate does not work with target only.
        unitary_gate = UnitaryGate(unitaries[0])
        gate.append(unitary_gate, [target])

    return gate


def _update_isometry(iso, gate):
    # "gate" matrix representation. Updates isometry.
    gate_matrix = qi.Operator(gate).data
    iso[:, :] = gate_matrix @ iso


def _mc_unitary(iso, col_index, bit_index):
    col = iso[:, col_index]
    idx1 = 2 * _a(col_index, bit_index + 1) * 2**bit_index + _b(
        col_index, bit_index + 1
    )
    idx2 = (2 * _a(col_index, bit_index + 1) + 1) * 2**bit_index + _b(
        col_index, bit_index + 1
    )

    return _unitary([[col[idx1]], [col[idx2]]], basis=0)


def _uc_unitaries(iso, n_qubits, col_index, bit_index):
    start = _a(col_index, bit_index + 1) + 1
    if _b(col_index, bit_index + 1) == 0:
        start -= 1

    gates = []
    for i in range(start):
        gates.append(np.identity(2))

    col = iso[:, col_index]
    for i in range(start, 2 ** (n_qubits - bit_index - 1)):
        idx1 = 2 * i * 2**bit_index + _b(col_index, bit_index)
        idx2 = (2 * i + 1) * 2**bit_index + _b(col_index, bit_index)

        gates.append(
            _unitary([[col[idx1]], [col[idx2]]], basis=_k_s(col_index, bit_index))
        )

    return gates


def _unitary(iso, basis=0):  # Lemma2 of https://arxiv.org/abs/1501.06911
    iden = np.identity(2)
    iso_norm = np.linalg.norm(iso, axis=0)[0]

    if iso_norm != 0.0:
        psi = iso / iso_norm

        psi_dagger = np.conj(psi.T)

        phi = -psi_dagger.dot(iden[:, [1]]) * iden[:, [0]]
        phi = phi + psi_dagger.dot(iden[:, [0]]) * iden[:, [1]]
        phi_dagger = np.conj(phi.T)

        unitary = np.kron(iden[:, [basis]], psi_dagger)
        unitary = unitary + np.kron(iden[:, [-(basis + 1)]], phi_dagger)

        return unitary

    return iden


def _a(col_index, bit_index):  # col_index >> bit_index.
    """
    Returns int representing n-bit_index most significant bits.
    """
    return col_index // 2**bit_index


def _b(col_index, bit_index):  # col_index ^ ((col_index >> bit_index) << bit_index)
    """
    Returns int representing bit_index less significant bits.
    """
    return col_index - (_a(col_index, bit_index) * 2**bit_index)


def _k_s(col_index, bit_index):
    # Returns the bit value at bit_index of col_index (k in the paper).
    return (col_index & 2**bit_index) // 2**bit_index


def cnot_count(isometry, scheme="ccd", method="estimate"):
    """
    Count the number of CNOTs to decompose the isometry.
    """
    if method == "estimate":
        return _cnot_count_estimate(isometry, scheme)

    # Exact count
    circuit = decompose(isometry, scheme)
    transpiled_circuit = transpile(
        circuit, basis_gates=["u", "cx"], optimization_level=0
    )
    count_ops = transpiled_circuit.count_ops()
    if "cx" in count_ops:
        return count_ops["cx"]

    return 0


def _cnot_count_estimate(isometry, scheme="ccd"):
    """
    Estimate the number of CNOTs to decompose the isometry.
    """
    iso = isometry.astype(complex)
    if len(iso.shape) == 1:
        iso = iso.reshape(iso.shape[0], 1)

    log_lines = int(log2(iso.shape[0]))
    log_cols = int(log2(iso.shape[1]))

    if scheme == "knill":
        return _cnot_count_estimate_knill(isometry, log_lines, log_cols)

    if scheme == "csd":
        return _cnot_count_estimate_csd(isometry, log_lines, log_cols)

    # CCD
    return _cnot_count_estimate_ccd(log_lines, log_cols)


def _cnot_count_estimate_csd(iso, log_lines, log_cols):
    unitary_gate = _extend_to_unitary(iso, log_lines, log_cols)

    return unitary_cnot_count(
        unitary_gate, decomposition="qsd", iso=log_lines - log_cols, apply_a2=True
    )


def _cnot_count_estimate_knill(iso, log_lines, log_cols):
    """
    Estimate the number of CNOTs to decompose the isometry using Knill.
    """
    unitary = _extend_to_unitary(iso, log_lines, log_cols)

    eigval, eigvec = np.linalg.eig(unitary)
    arg = np.angle(eigval)

    # pylint: disable=import-outside-toplevel
    from qclib.state_preparation.lowrank import cnot_count as schmidt_cnot_count

    cnots = 0
    for i in range(2**log_lines):
        if np.abs(arg[i]) > 10**-7:
            state = eigvec[:, i]

            # Two times Schmidt state preparation
            cnots += 2 * schmidt_cnot_count(state)

            # MCP
            circuit = QuantumCircuit(log_lines)
            circuit.mcp(3.0, list(range(log_lines - 1)), log_lines - 1)
            transpiled_circuit = transpile(
                circuit, basis_gates=["u", "cx"], optimization_level=0
            )
            if "cx" in transpiled_circuit.count_ops():
                cnots += transpiled_circuit.count_ops()["cx"]

    return cnots


def _cnot_count_estimate_ccd(log_lines, log_cols):
    """
    Estimate the number of CNOTs to decompose the isometry using CCD.
    """
    cnots = 0
    for k in range(2**log_cols):
        k_bin = f"{k:0{log_lines}b}"
        # G_K
        for i in range(log_lines):
            target = log_lines - i - 1
            control = list(range(target))
            ancilla = list(range(target + 1, log_lines))

            if _k_s(k, i) == 0 and _b(k, i + 1) != 0:
                # MCG implemented as a UCG up to a diagonal
                n_qubits = sum(k_bin[q] == "1" for q in control + ancilla) + 1
                cnots += 2 ** (n_qubits - 1) - 1

            # UCG up to a diagonal
            n_qubits = len(control) + 1
            cnots += 2 ** (n_qubits - 1) - 1

    # Diagonal
    if log_cols > 0:
        cnots += 2**log_cols - 2

    return cnots
