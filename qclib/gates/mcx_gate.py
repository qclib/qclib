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
    Implementation of an n-qubit controlled X gate as shown by He et al. (2017) in
        https://link.springer.com/10.1007/s10773-017-3389-4
    without using approximate Toffoli gates.
"""
import qiskit
import numpy as np


def cnx(circuit: qiskit.QuantumCircuit, controls: qiskit.QuantumRegister, free: list, targ: int):
    """
    Parameters
    ----------
    circuit: qiskit.QuantumCircuit
    controls: qiskit.QuantumRegister
    free: list of free qubits
    targ: target qubit
    Returns
    -------
    """

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
