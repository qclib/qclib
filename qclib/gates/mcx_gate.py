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


def _mcx_no_ancilla(circuit, controls, free, targ):
    n_controls = len(controls)
    n_free = len(free)
    targ = [targ] + free[::-1]

    for k in range(2):
        for i, _ in enumerate(controls):
            if i < n_controls - 2:
                circuit.ccx(control_qubit1=controls[n_controls - i - 1], control_qubit2=free[n_free - i - 1], target_qubit=targ[i])
            else:
                circuit.ccx(control_qubit1=controls[n_controls - i - 2], control_qubit2=controls[n_controls - i - 1], target_qubit=targ[i])
                
                break
        
        for i, _ in enumerate(free[1:]):
            circuit.ccx(control_qubit1=controls[2 + i], control_qubit2=free[i], target_qubit=free[i + 1])

    return


def mcx(n_qubits: int):
    """
    Parameters
    ----------
    n_qubits: int

    Returns
    -------
    quantum circuit implementing mcx with ancilla
    """

    k1 = int(np.ceil(n_qubits / 2))
    k2 = int(np.floor(n_qubits / 2)) - 1

    qr_k1_controls = QuantumRegister(k1, 'k1_control')
    qr_k2_controls = QuantumRegister(k2, 'k2_control')
    qr_target = QuantumRegister(1, 'target')
    qr_ancilla = QuantumRegister(1, 'ancilla')
    n_free_k1 = k1 - 2
    n_free_k2 = k2 + 1 - 2
    qr_free_k1 = list(range(n_qubits - n_free_k1, n_qubits))
    qr_free_k2 = list(range(n_qubits - k2 - n_free_k2 - 1, n_qubits - k2 - 1))
    qr_k2_plus_one_controls = list(range(k1, k1 + k2 + 1))

    qc_mcx = QuantumCircuit(qr_k1_controls, qr_k2_controls, qr_target, qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k1_controls, free=qr_free_k1, targ=qr_ancilla)

    qc_mcx.h(qr_target)
    qc_mcx.s(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k2_plus_one_controls[::-1], free=qr_free_k2[::-1], targ=qr_ancilla)

    qc_mcx.sdg(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k1_controls, free=qr_free_k1, targ=qr_ancilla)

    qc_mcx.s(qr_ancilla)

    mcx_no_ancilla(circuit=qc_mcx, controls=qr_k2_plus_one_controls[::-1], free=qr_free_k2[::-1], targ=qr_ancilla)

    qc_mcx.h(qr_target)
    qc_mcx.sdg(qr_ancilla)

    return qc_mcx
