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
Implements the state preparation
defined at https://link.springer.com/article/10.1007/s11128-010-0177-y
"""

from math import log2, pi

import numpy as np
from sympy import symbols, Or, And, Not
from sympy.logic.boolalg import simplify_logic
from qiskit import QuantumCircuit
from qiskit.circuit.library import RYGate
from qiskit.quantum_info import Operator
from qclib.gates.initialize import Initialize
from qclib.gates import Mcg
from qclib.gates.ucr import ucr

# pylint: disable=maybe-no-member

class FrqiInitialize(Initialize):
    """
    A flexible representation of quantum images for polynomial
    preparation, image compression, and processing operations
    https://link.springer.com/article/10.1007/s11128-010-0177-y

    This class implements a state preparation gate.
    """

    def __init__(self, params, label=None, opt_params=None):
        """
        Parameters
        ----------
        params: list of angles
            A vector representing an image.
            Values are angles representing color.

        opt_params: {'rescale': rescale
                     'method': method}
            rescale: bool
                If `True`, it rescales the values of the `params`
                vector to the range between 0 and pi.
                Default is ``rescale=False``.
            method: method
                Scheme used to decompose uniformed controlled rotations.
                Possible values are ``'ucr'`` (multiplexer), ``'mcr'``
                (multicontrolled rotations), and ``'auto'``.
                Default is ``method='auto'``.
        """
        self._name = "frqi"

        if opt_params is None:
            self.rescale = False
            self.method = 'auto'
        else:
            self.rescale = False if opt_params.get("rescale") is None \
                                    else opt_params.get("rescale")
            self.method = 'auto' if opt_params.get("method") is None \
                                    else opt_params.get("method")

        scaled_params = params
        if self.rescale:
            scaled_params = (
                (np.array(params) - np.min(params)) /
                (np.max(params) - np.min(params)) * pi
            )

        self._get_num_qubits(scaled_params)

        if label is None:
            label = "FRQI"

        super().__init__(self._name, self.num_qubits, scaled_params, label=label)

    def validate_parameter(self, parameter):
        if isinstance(parameter, (int, float)):
            return float(parameter)
        if isinstance(parameter, np.number):
            return float(parameter.item())

        raise TypeError(
            f"invalid param type {type(parameter)} for instruction {self.name}."
        )

    def _get_num_qubits(self, params):
        self.num_qubits = log2(len(params))

        # Check if param is a power of 2
        if self.num_qubits == 0 or not self.num_qubits.is_integer():
            raise ValueError("The length of the state vector is not a positive power of 2.")

        # Check if any pixels values is not between 0 and pi/2
        if any(0 > x > pi for x in params):
            raise ValueError("All pixel values must be between 0 and pi/2.")

        self.num_qubits = int(self.num_qubits) + 1

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        circuit = QuantumCircuit(self.num_qubits)
        circuit.h(circuit.qubits[:-1])

        simplified = {}
        if self.method in ('mcg', 'auto'):
            groups = self._group_binary_strings(self.params)
            for k, v in groups.items():
                simplified[k] = self._simplify_logic(v)

        mcg_cnot_count = 0
        if self.method == 'auto':
            for k, v in groups.items():
                for binary_string in simplified[k]:
                    indexes, ctrl_state = self._ctrl_state(binary_string)
                    n = len(indexes)
                    if n == 2:
                        mcg_cnot_count += 2
                    elif n == 3:
                        mcg_cnot_count += 4
                    else:
                        mcg_cnot_count += 16*n-40

        if self.method == 'ucr' or 2**self.num_qubits-1 < mcg_cnot_count:
            circuit.compose(
                ucr(RYGate, self.params),
                circuit.qubits[::-1],
                inplace=True
            )
        else:
            for k, v in groups.items():
                gate_matrix = Operator(RYGate(k)).data
                for binary_string in simplified[k]:
                    indexes, ctrl_state = self._ctrl_state(binary_string)
                    mcg = Mcg(
                        gate_matrix,
                        len(indexes),
                        ctrl_state=ctrl_state
                    )
                    circuit.compose(
                        mcg,
                        [*indexes, self.num_qubits-1],
                        inplace=True
                    )
        return circuit

    @staticmethod
    def initialize(circuit, state, qubits=None, opt_params=None):
        """
        Appends a FrqiInitialize gate into the circuit
        """
        if qubits is None:
            circuit.append(
                FrqiInitialize(state, opt_params=opt_params), circuit.qubits
            )
        else:
            circuit.append(
                FrqiInitialize(state, opt_params=opt_params), qubits
            )


    @staticmethod
    def _ctrl_state(binary_string):
        indexes = []
        ctrl_state = []
        for i, b in enumerate(binary_string):
            if b != '-':
                indexes.append(i)
                ctrl_state.append(b)

        return indexes, ''.join(ctrl_state)

    @staticmethod
    def _group_binary_strings(values):
        groups = {}

        n = int(log2(len(values)))

        for i, value in enumerate(values):
            binary_string = f'{i:{n}b}'[::-1]

            key = None
            for k in groups:
                if np.isclose(value, k):
                    key = k
                    break
            if key is not None:
                groups[key].append(binary_string)
            else:
                groups[value] = [binary_string]

        return groups

    @staticmethod
    def _simplify_logic(binary_strings):
        # Step 1: Define the number of variables
        n = len(binary_strings[0])

        # Step 2: Create a vector of symbolic variables dynamically
        variables = symbols(f'x0:{n}')

        # Step 3: Convert each binary string into a logical expression
        def binary_string_to_expression(binary_str, variables):
            terms = []
            for i, bit in enumerate(binary_str):
                if bit == '1':
                    terms.append(variables[i])  # Add the variable directly for '1'
                else:
                    terms.append(Not(variables[i]))  # Add the negation for '0'
            return And(*terms)  # Return the conjunction (AND) of terms

        # Convert each binary string into a logical expression
        expressions = [
            binary_string_to_expression(bin_str, variables)
            for bin_str in binary_strings
        ]

        # Step 4: Sum (OR) the logical expressions
        summation_expr = Or(*expressions)

        # Step 5: Simplify the Boolean expression using SymPy
        simplified_expr = simplify_logic(summation_expr, form='dnf')

        # Step 6: Convert the simplified expression back into binary strings
        def expression_to_binary_strings(simplified_expr, variables):
            binary_strings = []

            # Handle cases where the simplified expression is a single term
            if isinstance(simplified_expr, And):
                simplified_expr = [simplified_expr]
            else:
                simplified_expr = simplified_expr.args

            # Iterate over each term (conjunction) in the simplified expression
            for term in simplified_expr:
                binary_string = ['-'] * n  # Initialize binary string with don't-cares

                # Handle the case of single terms without Or
                if not isinstance(term, And):
                    term = [term]
                else:
                    term = term.args

                # Iterate over each literal in the term
                for literal in term: # .args if isinstance(term, And) else [term]:
                    if isinstance(literal, Not):  # If it's negated
                        variable = literal.args[0]  # Get the variable inside Not
                        idx = variables.index(variable)
                        binary_string[idx] = '0'
                    else:
                        idx = variables.index(literal)
                        binary_string[idx] = '1'

                binary_strings.append("".join(binary_string))

            return binary_strings

        # Step 7: Output the result
        binary_strings_output = expression_to_binary_strings(simplified_expr, variables)

        return binary_strings_output
