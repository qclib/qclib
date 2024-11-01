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
from qiskit import QuantumCircuit, QuantumRegister
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
            method: str
                Scheme used to decompose uniformed controlled rotations.
                Possible values are ``'ucr'`` (multiplexer), ``'mcr'``
                (multicontrolled rotations), and ``'auto'``.
                Default is ``method='auto'``.
            init_index_register: bool
                Specifies whether the index register should be
                initialized. This is achieved by applying a layer of
                Hadamard gates to the control qubits.
                Default is ``init_index_register=True``.
            simplify: bool
                Enables a search to minimize the use of controls, thereby
                reducing circuit cost. Note that this feature may incur a
                high classical cost for longer inputs.
                The default value is ``simplify=True``.
            separability: bool
                This parameter enables the search for separability
                in the state produced by the decomposition, potentially
                reducing the circuit cost at the expense of increased
                classical computation. It is only applicable when
                `simplify=True`.
                The default setting is ``separability=True``.
        """
        self._name = "frqi"

        if opt_params is None:
            self.rescale = False
            self.method = 'auto'
            self.init_index_register = True
            self.simplify = True
            self.separability = True
        else:
            self.rescale = False \
                if opt_params.get("rescale") is None \
                    else opt_params.get("rescale")
            self.method = 'auto' \
                if opt_params.get("method") is None \
                    else opt_params.get("method")
            self.init_index_register = True \
                if opt_params.get("init_index_register") is None \
                    else opt_params.get("init_index_register")
            self.simplify = True \
                if opt_params.get("simplify") is None \
                    else opt_params.get("simplify")
            self.separability = True \
                if opt_params.get("separability") is None \
                    else opt_params.get("separability")

        scaled_params = params
        if self.rescale:
            scaled_params = (
                (np.array(params) - np.min(params)) /
                (np.max(params) - np.min(params)) * pi
            )

        self._get_num_qubits(scaled_params)

        self.controls = QuantumRegister(self.num_qubits-1)
        self.target = QuantumRegister(1)

        if label is None:
            label = "FRQI"

        super().__init__(
            self._name,
            self.num_qubits,
            scaled_params,
            label=label
        )

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
            raise ValueError(
                "The length of the state vector is not a positive power of 2."
            )

        # Check if any pixels values is not between 0 and pi/2
        if any(0 > x > pi for x in params):
            raise ValueError("All pixel values must be between 0 and pi/2.")

        self.num_qubits = int(self.num_qubits) + 1

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        num_controls = len(self.controls)

        # Collects data for the decomposition.
        idx_list = {}
        ctrl_state_list = {}
        groups = self._group_binary_strings(self.params)

        # Performs simplification.
        idx_list, ctrl_state_list = self._ctrl_states(groups)

        # Search for state separability.
        missing_idx = []
        if self.simplify and self.separability:
            # Returns the number of qubits that can be ignored,
            # reducing the length of the control register.
            # num_controls -= self._search_separability(idx_list, num_controls)
            missing_idx = self._search_separability(idx_list, num_controls)
            num_controls -= len(missing_idx)

        # Estimates the cost of a MCG decomposition to
        # autoselect between `ucr`and `mcg`.
        mcg_cnot_count = 0
        if self.method == 'auto':
            for k, v in groups.items():
                for bin_str in v:
                    idx = idx_list[bin_str]
                    n = len(idx)
                    if n == 2:
                        mcg_cnot_count += 2
                    elif n == 3:
                        mcg_cnot_count += 4
                    else:
                        mcg_cnot_count += 16*n-40

        # Constructs the quantum circuit.
        circuit = QuantumCircuit(self.controls, self.target)

        if self.init_index_register:
            circuit.h(self.controls)

        if self.method == 'ucr' or (self.method == 'auto' and 2**num_controls < mcg_cnot_count):
            params = self.params
            controls = self.controls

            if self.simplify and self.separability and num_controls < len(self.controls):
                angles = {}
                controls = self.complement(len(self.controls), missing_idx)[::-1]

                for k, v in groups.items():
                    for bin_str in v:
                        angles[ctrl_state_list[bin_str]] = k

                for i in range(2**num_controls):
                    bin_str = f'{i:0{num_controls}b}'
                    if bin_str not in angles:
                        angles[bin_str] = 0.0

                angles = dict(sorted(angles.items()))
                params = list(angles.values())

            # `ucr` qubit index 0 is the target.
            circuit.compose(
                ucr(RYGate, params),
                [*self.target, *controls],
                inplace=True
            )
        else:
            for k, v in groups.items():
                gate_matrix = Operator(RYGate(k)).data
                for bin_str in v:
                    idx = idx_list[bin_str]
                    ctrl_state = ctrl_state_list[bin_str]
                    mcg = Mcg(
                        gate_matrix,
                        len(idx),
                        ctrl_state=ctrl_state
                    )
                    circuit.compose(
                        mcg,
                        [*self.controls[idx], *self.target],
                        inplace=True
                    )

        return circuit

    @staticmethod
    def complement(length, indexes):
        """
        Returns the complement of an integer list.
        """
        complement = sorted(set(range(length)).difference(set(indexes)))
        return complement

    @staticmethod
    def initialize(q_circuit, state, qubits=None, opt_params=None):
        """
        Appends a FrqiInitialize gate into the circuit
        """
        if qubits is None:
            q_circuit.append(
                FrqiInitialize(
                    state,
                    opt_params=opt_params
                ).definition,
                q_circuit.qubits
            )
        else:
            q_circuit.append(
                FrqiInitialize(
                    state,
                    opt_params=opt_params
                ).definition,
                qubits
            )

    def _ctrl_states(self, groups):
        idx_list = {}
        ctrl_state_list = {}

        full_control_register = list(range(len(self.controls)))

        for value, binary_strings in groups.items():
            if self.simplify and len(binary_strings) > 1:
                groups[value] = self._simplify_logic(binary_strings)
                for binary_string in groups[value]:
                    idx_list[binary_string], ctrl_state_list[binary_string] = \
                        self._ctrl_state(binary_string)
            else:
                for binary_string in binary_strings:
                    idx_list[binary_string] = full_control_register
                    ctrl_state_list[binary_string] = binary_string

        return idx_list, ctrl_state_list

    @staticmethod
    def _search_separability(idx_list, num_controls):
        def missing_numbers(n, lists):
            all_numbers = set(num for lst in lists for num in lst)
            full_range = set(range(n))
            missing = full_range - all_numbers
            return sorted(missing)
        def are_all_same_length(lists):
            if len(lists) <= 1:
                return True
            first_length = len(lists[0])
            return all(len(lst) == first_length for lst in lists)
        missing_idx = missing_numbers(num_controls, list(idx_list.values()))
        if are_all_same_length(list(idx_list.values())):
            # If `len(missing_idx)>0`, the state is separable.
            # If `are_all_same_length==True`, it is possible to use `ucr`
            # over a reduced number of controls.
            return missing_idx

        return []

    @staticmethod
    def _ctrl_state(binary_string):
        indexes = []
        ctrl_state = []
        n = len(binary_string)
        for i, b in enumerate(binary_string):
            if b != '-':
                indexes.append(n-i-1)
                ctrl_state.append(b)

        return indexes, ''.join(ctrl_state)[::-1]

    @staticmethod
    def _group_binary_strings(values):
        groups = {}

        n = int(log2(len(values)))

        original_groups = dict(
            sorted(enumerate(values), key=lambda item: item[1])
        )
        last_value = float('inf')
        for i, value in original_groups.items():
            binary_string = f'{i:0{n}b}'

            if np.isclose(value, last_value):
                groups[last_value].append(binary_string)
            else:
                groups[value] = [binary_string]
                last_value = value

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
                    # Add the variable directly for '1'
                    terms.append(variables[i])
                else:
                    # Add the negation for '0'
                    terms.append(Not(variables[i]))
            return And(*terms)  # Return the conjunction (AND) of terms

        # Convert each binary string into a logical expression
        expressions = [
            binary_string_to_expression(bin_str, variables)
            for bin_str in binary_strings
        ]

        # Step 4: Sum (OR) the logical expressions
        summation_expr = Or(*expressions)

        # Step 5: Simplify the Boolean expression using SymPy
        simplified_expr = simplify_logic(summation_expr, form='dnf', force=True, deep=False)

        # Step 6: Convert the simplified expression back into binary strings
        def expression_to_binary_strings(simplified_expr, variables):
            binary_strings = []
            dontcares = ['-'] * len(variables)

            # Ensure we're working with a list of terms
            terms = simplified_expr.args \
                if not isinstance(simplified_expr, And) else [simplified_expr]

            for term in terms:
                binary_string = dontcares.copy()

                # Ensure each term is iterable (a single variable might not be in a list)
                literals = term.args if isinstance(term, And) else [term]

                for literal in literals:
                    variable = literal.args[0] if isinstance(literal, Not) else literal
                    idx = variables.index(variable)
                    binary_string[idx] = '0' if isinstance(literal, Not) else '1'

                binary_strings.append("".join(binary_string))

            return binary_strings

        # Step 7: Output the result
        binary_strings_output = \
            expression_to_binary_strings(simplified_expr, variables)

        return binary_strings_output
