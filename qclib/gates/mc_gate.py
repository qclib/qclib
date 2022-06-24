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
n-qubit controlled gate
"""
from collections import namedtuple
import qiskit
import numpy as np


def mc_gate(gate: np.ndarray, circuit: qiskit.QuantumCircuit, controls: list, targ: int):
    """

    Parameters
    ----------
    gate: 2 X 2 unitary gate
    circuit: qiskit.QuantumCircuit
    controls: list of control qubits
    targ: target qubit

    Returns
    -------

    """

    n_qubits = len(controls) + 1
    gate_circuit = qiskit.QuantumCircuit(n_qubits, name="T" + str(targ))
    gate_circuit.permutation = list(range(len(controls) + 1))
    _c1c2(gate, n_qubits, gate_circuit)
    _c1c2(gate, n_qubits, gate_circuit, step=-1)

    _c1c2(gate, n_qubits - 1, gate_circuit, False)
    _c1c2(gate, n_qubits - 1, gate_circuit, False, step=-1)

    circuit.compose(gate_circuit, controls + [targ], inplace=True)
    circuit.permutation = gate_circuit.permutation


def _c1c2(gate, n_qubits, qcirc, first=True, step=1):
    pairs = namedtuple("pairs", ["control", "target"])

    if step == 1:
        start = 0
        reverse = True
    else:
        start = 1
        reverse = False

    qubit_pairs = [pairs(control, target)
                   for target in range(n_qubits)
                   for control in range(start, target)]

    qubit_pairs.sort(key=lambda e: e.control+e.target, reverse=reverse)

    for pair in qubit_pairs:
        exponent = pair.target - pair.control
        if pair.control == 0:
            exponent = exponent - 1
        param = 2 ** exponent
        signal = -1 if (pair.control == 0 and not first) else 1
        signal = step * signal
        if pair.target == n_qubits - 1 and first:
            csqgate = _gate_u(gate, param, signal)
            qcirc.compose(csqgate, qubits=[pair.control, pair.target], inplace=True)
        else:
            qcirc.crx(signal * np.pi / param, pair.control, pair.target)


def _gate_u(agate, coef, signal):
    param = 1/np.abs(coef)

    values, vectors = np.linalg.eig(agate)
    gate = np.power(values[0]+0j, param) * vectors[:, [0]] @ vectors[:, [0]].conj().T
    gate = gate + np.power(values[1]+0j, param) * vectors[:, [1]] @ vectors[:, [1]].conj().T

    if signal < 0:
        gate = np.linalg.inv(gate)

    sqgate = qiskit.QuantumCircuit(1, name='U^1/' + str(coef))
    sqgate.unitary(gate, 0)  # pylint: disable=maybe-no-member
    csqgate = sqgate.control(1)

    return csqgate
