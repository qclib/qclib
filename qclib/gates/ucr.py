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
Constructs a multiplexor gate.
"""

from math import log2
from typing import List, Union, Type
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZGate, RYGate, CXGate, CZGate


def ucr(r_gate: Union[Type[RZGate], Type[RYGate]],
        angles: List[float],
        c_gate: Union[Type[CXGate], Type[CZGate]]=CXGate,
        last_control=True) -> QuantumCircuit:
    """
    Constructs a multiplexor rotation gate.

    Synthesis of Quantum Logic Circuits
    https://arxiv.org/abs/quant-ph/0406176
    """
    size = len(angles)
    n_qubits = int(log2(size)) + 1

    reg = QuantumRegister(n_qubits)
    circuit = QuantumCircuit(reg)

    target = reg[0]
    control = reg[n_qubits-1]

    if n_qubits == 1:
        if abs(angles[0]) > 10**-8:
            circuit.append(r_gate(angles[0]), [target])
        return circuit

    angle_multiplexor = np.kron([[0.5, 0.5], [0.5, -0.5]], np.identity(2**(n_qubits-2)))
    multiplexed_angles = angle_multiplexor.dot(angles)

    # Figure 2 from Synthesis of Quantum Logic Circuits:
    #   The recursive decomposition of a multiplexed Rz gate.
    #   The boxed CNOT gates may be canceled.
    # This is why "last_cnot=False" in both calls of "rotation_multiplexor()" and
    # also why the multiplexer in the second "circuit.append()" is reversed.
    mult = ucr(r_gate, multiplexed_angles[:size//2], c_gate, False)
    circuit.append(mult.to_instruction(), reg[0:-1])

    circuit.append(c_gate(), [control, target])

    mult = ucr(r_gate, multiplexed_angles[size//2:], c_gate, False)
    circuit.append(mult.reverse_ops().to_instruction(), reg[0:-1])

    # The following condition allows saving CNOTs when two multiplexors are used
    # in sequence. Any multiplexor can have its operation reversed. Therefore, if
    # the second multiplexor is reverted, its last CNOT will be cancelled by the
    # last CNOT of the first multiplexer. In this condition, both last CNOTs are
    # unnecessary.
    if last_control:
        circuit.append(c_gate(), [control, target])

    return circuit
