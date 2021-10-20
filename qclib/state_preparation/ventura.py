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
https://arxiv.org/abs/quant-ph/9807054
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister

# pylint: disable=maybe-no-member

def initialize(state, n_qubits, n_output_values):
    """ State preparation using Ventura and Martinez algorithm quant-ph/9807054
    Algorithm that requires a polynomial number of elementary operations for
    initializing a quantum system to represent only the m known points of a
    function f (m = len(state)).
    The result is a quantum superposition with m nonzero coefficients -- the
    creation of which is a nontrivial task compared to creating a superposition
    of all basis states.
    The amplitudes of modulus "1/sqrt(m)" will be "2 pi / N" radians apart from
    each other on the complex plane.

    Binary output function case:
        f:z->s with z \\in {0,1}^n and s \\in {0, 1}
    General case:
        f:z->s with z \\in {0,1}^n and s \\in {0, 1, ..., N-1}

    For instance, to initialize the state
    1/sqrt(3)|01> + 1/sqrt(3)*e^(1*i*2pi/N)|10> + 1/sqrt(3)*e^(2*i*2pi/N)c|11>
        $ state = {1: 0, 2: 1, 3: 2}
        $ circuit = initialize(state, n=2, N=3)

    Parameters
    ----------
    state: dict of {int:float}
        A unit vector representing a quantum state.
        Keys are function binary input values and values are function output values.

    n_qubits: int
        Number of bits to represent the binary values.

    n_output_values: int
        Number of possible output values N (Ex.: n_output_values=2 for a binary function).

    Returns
    -------
    circuit: QuantumCircuit
        QuantumCircuit to initialize the state.
    """

    reg_x = QuantumRegister(n_qubits, 'x')
    reg_g = QuantumRegister(n_qubits - 1, 'g')
    reg_c = QuantumRegister(2, 'c')

    circuit = QuantumCircuit(reg_x, reg_g, reg_c)

    reg_x = reg_x[::-1] # qiskit reverse (qiskit little-endian)

    bits_z0 = [int(k) for k in '{:0{}b}'.format(0, n_qubits)]
    for idx_p, (input_z, output_s) in list(enumerate(state.items()))[::-1]:
        bits_z = [int(k) for k in '{:0{}b}'.format(input_z, n_qubits)]

        circuit.x(reg_c[1])
        for j,_ in enumerate(bits_z):
            if bits_z0[j] != bits_z[j]:
                circuit.cx(reg_c[1], reg_x[j])

        bits_z0 = bits_z
        circuit.cx(reg_c[1], reg_c[0])
        circuit.x(reg_c[1])

        theta = -2*np.arccos(np.sqrt(idx_p/(idx_p+1))) # This sign is here for the smaller values
                                                       # of "s" to be represented by negative
                                                       # amplitudes and the larger ones by positive
                                                       # amplitudes.
        lamb = -output_s*2*np.pi/n_output_values # In the paper this negative sign is missing.
                                                 # Without it the matrix S is not unitary.
        phi = -lamb
        circuit.cu(theta, phi, lamb, 0, reg_c[0], reg_c[1])

        if bits_z[0] == 0:
            circuit.x(reg_x[0])
        if bits_z[1] == 0:
            circuit.x(reg_x[1])

        circuit.ccx(reg_x[0], reg_x[1], reg_g[0])

        if bits_z[0] == 0:
            circuit.x(reg_x[0])
        if bits_z[1] == 0:
            circuit.x(reg_x[1])

        for k in range(2, n_qubits):
            if bits_z[k] == 0:
                circuit.x(reg_x[k])

            circuit.ccx(reg_x[k], reg_g[k-2], reg_g[k-1])

            if bits_z[k] == 0:
                circuit.x(reg_x[k])

        circuit.cx(reg_g[n_qubits - 2], reg_c[0])

        for k in range(n_qubits - 1, 1, -1):
            if bits_z[k] == 0:
                circuit.x(reg_x[k])

            circuit.ccx(reg_x[k], reg_g[k-2], reg_g[k-1])

            if bits_z[k] == 0:
                circuit.x(reg_x[k])

        if bits_z[0] == 0:
            circuit.x(reg_x[0])
        if bits_z[1] == 0:
            circuit.x(reg_x[1])

        circuit.ccx(reg_x[0], reg_x[1], reg_g[0])

        if bits_z[0] == 0:
            circuit.x(reg_x[0])
        if bits_z[1] == 0:
            circuit.x(reg_x[1])

    circuit.x(reg_c[1])

    return circuit
