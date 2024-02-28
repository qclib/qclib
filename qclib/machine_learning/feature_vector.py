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

import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumRegister, ParameterVector, Instruction
from qiskit.circuit.library import BlueprintCircuit
from qclib.gates.initialize import Initialize

# pylint: disable=relative-beyond-top-level

class FeatureVector(BlueprintCircuit):
    """
    This circuit acts as parameterized initialization for statevectors with ``feature_dimension``
    dimensions. The circuit contains a placeholder instruction that can only be
    synthesized/defined when all parameters are bound.

    In ML, this circuit can be used to load the training data into qubit amplitudes. It does not
    apply an kernel transformation (therefore, it is a "raw" feature vector).

    This circuit can't be used with gradient based optimizers.

    Examples:

    .. code-block::

        from qclib.state_preparation import FeatureVector, BaaLowRankInitialize
        circuit = FeatureVector(2, 4, BaaLowRankInitialize, opt_params=None)
        print(circuit.num_qubits)
        # prints: 2

        print(circuit.draw(output='text'))
        # prints:
        #      ┌────────────────────────────────────────┐
        # q_0: ┤0                                              ├
        #      │  ParameterizedInitialize(x[0],x[1],x[2],x[3]) │
        # q_1: ┤1                                              ├
        #      └────────────────────────────────────────┘

        print(circuit.ordered_parameters)
        # prints: [Parameter(p[0]), Parameter(p[1]), Parameter(p[2]), Parameter(p[3])]

        import numpy as np
        state = np.array([1, 0, 0, 1]) / np.sqrt(2)
        bound = circuit.assign_parameters(state)
        print(bound.draw())
        # prints:
        #      ┌────────────────────────────────────────┐
        # q_0: ┤0                                              ├
        #      │  ParameterizedInitialize(0.70711,0,0,0.70711) │
        # q_1: ┤1                                              ├
        #      └────────────────────────────────────────┘

    """

    def __init__(self,
                 num_qubits: int,
                 feature_dimension: int,
                 initializer: Initialize,
                 opt_params = None) -> None:

        super().__init__()

        self._ordered_parameters = ParameterVector("x")

        self.num_qubits = num_qubits
        self._feature_dimension = None
        if feature_dimension is not None:
            self.feature_dimension = feature_dimension

        self._opt_params = opt_params
        self._initializer = initializer

    def _unsorted_parameters(self):
        if self.data is None:
            self._build()
        return super()._unsorted_parameters()

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        if self.num_qubits == 0:
            return

        placeholder = ParameterizedInitialize(self.num_qubits,
                                                self._ordered_parameters[:],
                                                self._initializer,
                                                self._opt_params)
        self.append(placeholder, self.qubits)

    def _check_configuration(self, raise_on_failure=True):
        if isinstance(self._ordered_parameters, ParameterVector):
            self._ordered_parameters.resize(self.feature_dimension)
        elif len(self._ordered_parameters) != self.feature_dimension:
            if raise_on_failure:
                raise ValueError("Mismatching number of parameters and feature dimension.")
            return False
        return True

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return super().num_qubits

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the n-local circuit.

        Args:
            The new number of qubits.
        """
        if self.num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self.qregs = []
            if num_qubits is not None and num_qubits > 0:
                self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def feature_dimension(self) -> int:
        """Return the feature dimension.

        Returns:
            The feature dimension, which is ``2 ** num_qubits``.
        """
        return self._feature_dimension

    @feature_dimension.setter
    def feature_dimension(self, feature_dimension: int) -> None:
        """Set the feature dimension.

        Args:
            feature_dimension: The new feature dimension. Must be a power of 2.

        Raises:
            ValueError: If ``feature_dimension`` is not a power of 2.
        """

        if self._feature_dimension != feature_dimension:
            #self._invalidate()
            self._feature_dimension = feature_dimension
            #self.qregs = []
            #if self.num_qubits is not None and self.num_qubits > 0:
            #    self.qregs = [QuantumRegister(self.num_qubits, name="q")]

class ParameterizedInitialize(Instruction):
    """A normalized parameterized initialize instruction."""

    def __init__(self, n_qubits, amplitudes, initializer, opt_params):
        num_qubits = n_qubits

        super().__init__("ParameterizedInitialize", int(num_qubits), 0, amplitudes)

        self._initializer = initializer
        self._opt_params = opt_params

    def _define(self):
        # cast ParameterExpressions that are fully bound to numbers
        cleaned_params = []
        for param in self.params:
            if len(param.parameters) == 0:
                cleaned_params.append(complex(param))
            else:
                raise QiskitError(
                    "Cannot define a ParameterizedInitialize with unbound parameters"
                )

        # normalize
        normalized = np.array(cleaned_params) / np.linalg.norm(cleaned_params)

        gate = self._initializer(normalized, opt_params=self._opt_params)

        self.definition = gate.definition
