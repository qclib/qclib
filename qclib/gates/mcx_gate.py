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
    Implementation of an n-qubit controlled X with ancilla gate as shown by He et al. (2017) in
        https://link.springer.com/10.1007/s10773-017-3389-4
    without using approximate Toffoli gates.
"""
import qiskit
import numpy as np


def mcx_no_ancilla(circuit: qiskit.QuantumCircuit, \
                   controls: qiskit.QuantumRegister, \
                   free: list, \
                   targ: int):
    """
    Parameters
    ----------
    circuit: qiskit.QuantumCircuit
    controls: qiskit.QuantumRegister
    free: list
    targ: int

    Returns
    -------
    quantum circuit implementing mcx with free qubits without ancilla
    """

    n_controls = len(controls)
    n_free = len(free)
    targ = [targ] + free[::-1]

    for _ in range(2):
        for i, _ in enumerate(controls):
            if i < n_controls - 2:
                circuit.ccx(control_qubit1=controls[n_controls - i - 1], \
                            control_qubit2=free[n_free - i - 1], \
                            target_qubit=targ[i])
            else:
                circuit.ccx(control_qubit1=controls[n_controls - i - 2], \
                            control_qubit2=controls[n_controls - i - 1], \
                            target_qubit=targ[i])

                break

        for i, _ in enumerate(free[1:]):
            circuit.ccx(control_qubit1=controls[2 + i], \
                        control_qubit2=free[i], \
                        target_qubit=free[i + 1])


def mcx(n_controls: int):
    """
    Parameters
    ----------
    n_controls: int

    Returns
    -------
    quantum circuit implementing decomposed mcx with ancilla
    """

    n_qubits = n_controls + 1
    k_1 = int(np.ceil(n_qubits / 2))
    k_2 = int(np.floor(n_qubits / 2)) - 1

    qr_k_1_controls = qiskit.QuantumRegister(k_1, 'k_1_control')
    qr_k_2_controls = qiskit.QuantumRegister(k_2, 'k_2_control')
    qr_target = qiskit.QuantumRegister(1, 'target')
    qr_ancilla = qiskit.QuantumRegister(1, 'ancilla')
    n_free_k_1 = k_1 - 2
    n_free_k_2 = (k_2 + 1) - 2
    qr_free_k_1 = list(range(n_qubits - n_free_k_1, n_qubits))
    qr_free_k_2 = list(range(n_qubits - k_2 - n_free_k_2 - 1, n_qubits - k_2 - 1))
    qr_k_2_plus_one_controls = list(range(k_1, k_1 + k_2 + 1))

    qc_mcx = qiskit.QuantumCircuit(qr_k_1_controls, qr_k_2_controls, qr_target, qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k_1_controls, free=qr_free_k_1, targ=qr_ancilla)

    qc_mcx.h(qr_target)
    qc_mcx.s(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, \
                   controls=qr_k_2_plus_one_controls[::-1], \
                   free=qr_free_k_2[::-1], \
                   targ=qr_ancilla)

    qc_mcx.sdg(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k_1_controls, free=qr_free_k_1, targ=qr_ancilla)

    qc_mcx.s(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, \
                   controls=qr_k_2_plus_one_controls[::-1], \
                   free=qr_free_k_2[::-1], \
                   targ=qr_ancilla)

    qc_mcx.h(qr_target)
    qc_mcx.sdg(qr_ancilla)

    return qc_mcx
