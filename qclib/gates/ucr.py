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


from math import log2, pi
from typing import List, Union, Type
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate
from qiskit.circuit.library import RZGate, RYGate, CXGate, CZGate

from sympy import symbols, Or, And, Not
from sympy.logic.boolalg import simplify_logic as sp_simplify_logic
from qclib.gates import Mcg

_mcg_cnot_count = {1:2, 2:4, 3:14, 4:24, 5:40, 6:56, 7:80}

def multiplexor(
    r_gate: Union[Type[RZGate], Type[RYGate]],
    angles: List[float],
    c_gate: Union[Type[CXGate], Type[CZGate]] = CXGate,
    last_control: bool = True,
) -> QuantumCircuit:
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
    control = reg[n_qubits - 1]

    if n_qubits == 1:
        if abs(angles[0]) > 10**-8:
            circuit.compose(r_gate(angles[0]), [target], inplace=True)
        return circuit

    angle_multiplexor = np.kron(
        [[0.5, 0.5], [0.5, -0.5]], np.identity(2 ** (n_qubits - 2))
    )
    multiplexed_angles = angle_multiplexor @ angles

    # Figure 2 from Synthesis of Quantum Logic Circuits:
    #   The recursive decomposition of a multiplexed Rz gate.
    #   The boxed CNOT gates may be canceled.
    # This is why "last_cnot=False" in both calls of "rotation_multiplexor()" and
    # also why the multiplexer in the second "circuit.append()" is reversed.
    mult = multiplexor(r_gate, multiplexed_angles[: size // 2], c_gate, False)
    circuit.compose(mult, reg[0:-1], inplace=True)

    circuit.compose(c_gate(), [control, target], inplace=True)

    mult = multiplexor(r_gate, multiplexed_angles[size // 2 :], c_gate, False)
    circuit.compose(mult.reverse_ops(), reg[0:-1], inplace=True)

    # The following condition allows saving CNOTs when two multiplexors are used
    # in sequence. Any multiplexor can have its operation reversed. Therefore, if
    # the second multiplexor is reverted, its last CNOT will be cancelled by the
    # last CNOT of the first multiplexer. In this condition, both last CNOTs are
    # unnecessary.
    if last_control:
        circuit.compose(c_gate(), [control, target], inplace=True)

    return circuit

class Ucr(Gate):
    """
    Builds a uniformly controlled rotation gate.
    """
    def __init__(
        self,
        angle_list: List[List[float]],
        r_gate: Union[Type[RZGate], Type[RYGate]] = RYGate,
        up_to_diagonal: bool = False,
        method: str = 'multiplexor',
        simplify: bool = False,
        label=None
    ):
        if method not in ['auto', 'multiplexor', 'mcg']:
            raise ValueError(
                f"Method '{method}' is not one of the valid options "
                "['auto', 'multiplexor', 'mcg']."
            )

        self._name = "ucr"
        self.r_gate = r_gate
        self.up_to_diagonal = up_to_diagonal
        self.method = method
        self.simplify = simplify

        self._get_num_qubits(angle_list)
        self.controls = QuantumRegister(self.num_qubits-1)
        self.target = QuantumRegister(1)

        if label is None:
            label = "UCR"

        super().__init__(
            self._name,
            self.num_qubits,
            angle_list,
            label=label
        )

    def _get_num_qubits(self, params):
        self.num_qubits = log2(len(params))

        # Check if param is a power of 2
        if self.num_qubits == 0 or not self.num_qubits.is_integer():
            raise ValueError(
                "The length of the angle list is not a positive power of 2."
            )

        if self.r_gate is RYGate:
            # Check if any pixels values is not between 0 and pi
            if any(0 > x > pi for x in params):
                raise ValueError("All angle values must be between 0 and pi.")
        else:
            # Check if any pixels values is not between 0 and 2*pi
            if any(0 > x > 2*pi for x in params):
                raise ValueError("All angle values must be between 0 and 2*pi.")

        self.num_qubits = int(self.num_qubits) + 1

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):
        circuit = QuantumCircuit(self.controls, self.target)

        if not self.simplify:
            if self.method != 'mcg':
                # 'auto' or 'multiplexor'.

                ucr = multiplexor(self.r_gate, self.params)
                # `multiplexor` qubit index 0 is the target.
                circuit.compose(
                    ucr,
                    [*self.target, *self.controls],
                    inplace=True
                )

            else:
                # 'mcg'.

                n = len(self.controls)

                for i, value in enumerate(self.params):
                    ctrl_state = f'{i:0{n}b}'
                    gate_matrix = self.r_gate(value).to_matrix()

                    mcg = Mcg(
                        gate_matrix,
                        n,
                        ctrl_state=ctrl_state
                    )
                    circuit.compose(
                        mcg,
                        [*self.controls, *self.target],
                        inplace=True
                    )

        else:
            num_controls = len(self.controls)

            # Collects data for the decomposition.
            groups = self._group_binary_strings(self.params)

            # Ignores the most repeated angle controls.
            # Find the item with the largest number of lists.
            global_angle = 0.0
            max_item = max(groups.items(), key=lambda item: len(item[1]))
            global_angle = max_item[0]
            groups[global_angle] = []

            # Performs simplification.
            indexes, ctrl_states = self._ctrl_states(groups)

            # Search for separability (qubits not used after simplification).
            # Returns the number of qubits that can be ignored,
            # reducing the length of the control register.
            missing_indexes = self._search_separability(indexes, num_controls)
            num_controls -= len(missing_indexes)

            # Estimates the cost of a MCG decomposition to
            # autoselect between `multiplexor`and `mcg`.
            mcg_cnot_count = 2**num_controls + 1
            if self.method == 'auto':
                mcg_cnot_count = 0
                for k, v in groups.items():
                    for bin_str in v:
                        idx = indexes[bin_str]
                        n_controls = len(idx)
                        if n_controls < 8:
                            mcg_cnot_count += _mcg_cnot_count[n_controls]
                        else:
                            mcg_cnot_count += 16*(n_controls+1)-40

            if global_angle != 0.0:
                r_gate = self.r_gate(global_angle)
                circuit.compose(
                    r_gate,
                    self.target,
                    inplace=True
                )

            if self.method == 'multiplexor' or (
                self.method == 'auto' and 2**num_controls < mcg_cnot_count
            ):
                if len(missing_indexes) == 0:
                    # It is not possible to separate the state.
                    params = np.array(self.params) - global_angle
                    controls = self.controls
                else:
                    # It is possible to separate the state.
                    #
                    # Reduces the number of control bits, eliminating ignored
                    # indexes, and assembles an ordered and reduced list of
                    # angles according to the active bit patterns corresponding
                    # to each angle value.
                    angles = {}
                    controls = self.complement(len(self.controls), missing_indexes)

                    for k, v in groups.items():
                        for bin_str in v:
                            reduced_bin_str = \
                                [
                                    char for i, char in enumerate(bin_str[::-1]) if i in controls
                                ][::-1]
                            dontcares_indexes = \
                                [
                                    i for i, char in enumerate(reduced_bin_str) if char == '-'
                                ]
                            n_placeholders = len(dontcares_indexes)
                            for i in range(2**n_placeholders):
                                pattern = f'{i:0{n_placeholders}b}'
                                for j, dontcare_index in enumerate(dontcares_indexes):
                                    reduced_bin_str[dontcare_index] = pattern[j]
                                angles["".join(reduced_bin_str)] = k - global_angle

                    # Completes the positions of the missing bit patterns
                    # (if any) with zeros.
                    if len(angles) < 2**num_controls:
                        for i in range(2**num_controls):
                            bin_str = f'{i:0{num_controls}b}'
                            if bin_str not in angles:
                                angles[bin_str] = 0.0

                    angles = dict(sorted(angles.items()))
                    params = list(angles.values())

                # `multiplexor` qubit index 0 is the target.
                ucr = multiplexor(self.r_gate, params)
                circuit.compose(
                    ucr,
                    [*self.target, *controls],
                    inplace=True
                )

            else:
                # 'mcg'
                for k, v in groups.items():
                    gate_matrix = self.r_gate(k - global_angle).to_matrix()

                    for bin_str in v:
                        idx = indexes[bin_str]
                        ctrl_state = ctrl_states[bin_str]
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

    def _ctrl_states(self, groups):
        indexes = {}
        ctrl_states = {}

        full_control_register = list(range(len(self.controls)))

        for value, binary_strings in groups.items():
            if len(binary_strings) > 1:
                groups[value] = simplify_logic(binary_strings)
                for binary_string in groups[value]:
                    indexes[binary_string], \
                    ctrl_states[binary_string] = \
                        self._ctrl_state(binary_string)
            else:
                # Note that zero binary strings does not produce a result, as
                # expected, possibly allowing separation of the multiplexor
                # or reduction in the number of UCGs.
                for binary_string in binary_strings:
                    indexes[binary_string] = full_control_register
                    ctrl_states[binary_string] = binary_string

        return indexes, ctrl_states

    @staticmethod
    def _ctrl_state(binary_string):
        indexes = []
        ctrl_state = []
        n = len(binary_string)
        for i, b in enumerate(binary_string):
            if b != '-':
                # Indexes are inverted on Qiskit circuits.
                indexes.append(n-i-1)
                ctrl_state.append(b)
        # Sorts indexes in ascending order.
        return indexes[::-1], ''.join(ctrl_state)

    @staticmethod
    def complement(length, indexes):
        """
        Returns the complement of an integer list.
        """
        complement = sorted(set(range(length)).difference(set(indexes)))
        return complement

    @staticmethod
    def _search_separability(idx_list, num_controls):
        """
        If `len(missing)>0`, the `multiplexor` is separable.
        That is, it is possible to use `multiplexor` over a
        reduced number of controls.
        """
        def list_missing(n, lists):
            all_numbers = set(num for lst in lists for num in lst)
            full_range = set(range(n))
            missing = full_range - all_numbers
            return sorted(missing)

        missing = list_missing(num_controls, list(idx_list.values()))
        return missing

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

            if np.isclose(value, last_value, atol=0.0, rtol=1e-07):
                groups[last_value].append(binary_string)
            else:
                groups[value] = [binary_string]
                last_value = value

        return groups

def simplify_logic(binary_strings):
    """
    Convert each binary string into a logical expression
    expressions, sum (OR) the logical expressions, and
    simplify the boolean expression using SymPy.
    """
    # Step 0: Nothing to do
    if len(binary_strings) <= 1:
        return binary_strings

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
    simplified_expr = sp_simplify_logic(summation_expr, form='dnf', force=True, deep=False)

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
