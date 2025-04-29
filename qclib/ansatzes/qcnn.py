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

"""The Quantum Convolutional Neural Network (QCNN) circuit class.

    https://www.nature.com/articles/s41567-019-0648-8
    https://link.springer.com/article/10.1007/s42484-021-00061-x
"""

from __future__ import annotations

from typing import Sequence, Mapping

import numpy
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit._accelerate.circuit import QuantumRegister
from qiskit.circuit import Parameter, ParameterVector, ParameterExpression

from qiskit.circuit.library import BlueprintCircuit

"""List of convolutional and pooling circuit block templates.

    Different types of block templates can be mixed.
    For example, ConvDefault with PoolSimple.
"""

class BlockTemplate():
    """Base class for circuit block templates."""

    @property
    def num_parameters(self) -> int:
        raise ValueError("Property num_parameters not implemented.")

    @staticmethod
    def circuit(params):
        raise ValueError("Method circuit not implemented.")

"""Convolution templates"""

class ConvDefault(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 3

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.rz(-numpy.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)
        target.cx(1, 0)
        target.rz(numpy.pi / 2, 0)

        return target

class ConvSimple(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 2

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.ry(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)

        return target

class ConvU4(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 15

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.u(params[0], params[1], params[2], 0)
        target.u(params[3], params[4], params[5], 1)
        target.cx(1, 0)
        target.rz(params[6], 0)
        target.ry(params[7], 1)
        target.cx(0, 1)
        target.ry(params[8], 1)
        target.cx(1, 0)
        target.u(params[9], params[10], params[11], 0)
        target.u(params[12], params[13], params[14], 1)

        return target

class ConvU4UpToDiagonal(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 14

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.u(params[0], params[1], params[2], 0)
        target.u(params[3], params[4], params[5], 1)
        target.cx(0, 1)
        target.ry(params[6], 0)
        target.ry(params[7], 1)
        target.cx(0, 1)
        target.u(params[8], params[9], params[10], 0)
        target.u(params[11], params[12], params[13], 1)

        return target

"""Pooling templates"""

class PoolDefault(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 3

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.rz(-numpy.pi / 2, 1)
        target.cx(1, 0)
        target.rz(params[0], 0)
        target.ry(params[1], 1)
        target.cx(0, 1)
        target.ry(params[2], 1)

        return target

class PoolSimple(BlockTemplate):
    @property
    def num_parameters(self) -> int:
        return 2

    @staticmethod
    def circuit(params):
        target = QuantumCircuit(2)
        target.crz(params[0], 0, 1, ctrl_state='1')
        target.crx(params[1], 0, 1, ctrl_state='0')

        return target



class Qcnn(BlueprintCircuit):
    """The QCNN circuit class.

        TODO: documentation
    """

    def __init__(
        self,
        num_qubits: int | None = None,
        reps: int = 1,
        insert_barriers: bool = False,
        parameter_prefix: str = "θ",
        initial_state: QuantumCircuit | None = None,
        name: str | None = "qcnn",
        conv_block: BlockTemplate = ConvDefault(),
        pool_block: BlockTemplate = PoolDefault()
    ) -> None:
        """Create a new QCNN circuit.

        Args:
            num_qubits: The number of qubits of the circuit.
            reps: Specifies how often the rotation blocks and entanglement blocks are repeated.
            insert_barriers: If ``True``, barriers are inserted in between each layer. If ``False``,
                no barriers are inserted.
            parameter_prefix: The prefix used if default parameters are generated.
            initial_state: A :class:`.QuantumCircuit` object which can be used to describe an initial
                state prepended to the circuit.
            name: The name of the circuit.
            conv_layer: Circuit block for the convolutional layer.
            pool_layer: Circuit block for the pooling layer.

        Examples:
            from qclib.ansatzes.qcnn import (
                Qcnn,
                ConvDefault,
                ConvSimple,
                PoolDefault,
                PoolSimple
            )

            num_qubits = 4

            ansatz = Qcnn(
                num_qubits,
                conv_block=ConvSimple(),
                pool_block=PoolSimple(),
                insert_barriers=True
            )
            ansatz.decompose(reps=1).draw()

            # Output:
                 ┌─────────────────────────────────────────────────────────┐»
            q_0: ┤0                                                        ├»
                 │                                                         │»
            q_1: ┤1                                                        ├»
                 │  Convolutional(θ[0],θ[1],θ[2],θ[3],θ[4],θ[5],θ[6],θ[7]) │»
            q_2: ┤2                                                        ├»
                 │                                                         │»
            q_3: ┤3                                                        ├»
                 └─────────────────────────────────────────────────────────┘»
            «     ┌─────────────────────────────────┐                               »
            «q_0: ┤0                                ├───────────────────────────────»
            «     │                                 │                               »
            «q_1: ┤1                                ├───────────────────────────────»
            «     │  Pooling(θ[8],θ[9],θ[10],θ[11]) │┌─────────────────────────────┐»
            «q_2: ┤2                                ├┤0                            ├»
            «     │                                 ││  Convolutional(θ[12],θ[13]) │»
            «q_3: ┤3                                ├┤1                            ├»
            «     └─────────────────────────────────┘└─────────────────────────────┘»
            «
            «q_0: ─────────────────────────
            «
            «q_1: ─────────────────────────
            «     ┌───────────────────────┐
            «q_2: ┤0                      ├
            «     │  Pooling(θ[14],θ[15]) │
            «q_3: ┤1                      ├
            «     └───────────────────────┘

        Raises:
            ValueError: If ``reps`` parameter is less than or equal to 0.
            TypeError: If ``reps`` parameter is not an int value.
            TypeError: If ``conv_block`` parameter is not an BlockTemplate value.
            TypeError: If ``pool_block`` parameter is not an BlockTemplate value.
        """
        super().__init__(name=name)

        self._num_qubits: int | None = None
        self._insert_barriers = insert_barriers
        self._reps = reps
        self._parameter_prefix=parameter_prefix
        self._initial_state: QuantumCircuit | None = None
        self._initial_state_circuit: QuantumCircuit | None = None
        self._bounds: list[tuple[float | None, float | None]] | None = None

        if int(reps) != reps:
            raise TypeError("The value of reps should be int.")

        if reps < 1:
            raise ValueError("The value of reps should be larger than or equal to 1.")

        if not issubclass(type(conv_block), BlockTemplate):
            raise TypeError("The value of conv_block should be BlockTemplate.")

        if not issubclass(type(pool_block), BlockTemplate):
            raise TypeError("The value of pool_block should be BlockTemplate.")


        if num_qubits is not None:
            self.num_qubits = num_qubits

        if initial_state is not None:
            self.initial_state = initial_state

        self._conv_template = conv_block.circuit
        self._conv_template_num_parameters = conv_block.num_parameters

        self._pool_template = pool_block.circuit
        self._pool_template_num_parameters = pool_block.num_parameters


    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in this circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits if self._num_qubits is not None else 0

    @num_qubits.setter
    def num_qubits(self, num_qubits: int) -> None:
        """Set the number of qubits for the qcnn circuit.

        Args:
            The new number of qubits.
        """
        if self._num_qubits != num_qubits:
            # invalidate the circuit
            self._invalidate()
            self._num_qubits = num_qubits
            self.qregs = [QuantumRegister(num_qubits, name="q")]

    @property
    def num_layers(self) -> int:
        """Return the number of layers in the circuit.

        Returns:
            The number of layers in the circuit.
        """

        # A pool and a convolutional layer per iteration.
        # Multiply all by the number of repetitions.
        return self.reps * (2 * numpy.ceil(numpy.log2(self.num_qubits)))

    @property
    def insert_barriers(self) -> bool:
        """If barriers are inserted in between the layers or not.

        Returns:
            ``True``, if barriers are inserted in between the layers, ``False`` if not.
        """
        return self._insert_barriers

    @insert_barriers.setter
    def insert_barriers(self, insert_barriers: bool) -> None:
        """Specify whether barriers should be inserted in between the layers or not.

        Args:
            insert_barriers: If True, barriers are inserted, if False not.
        """
        # if insert_barriers changes, we have to invalidate the circuit definition,
        # if it is the same as before we can leave the instance as it is
        if insert_barriers is not self._insert_barriers:
            self._invalidate()
            self._insert_barriers = insert_barriers

    @property
    def reps(self) -> int:
        """The number of times the circuit is repeated.

        Returns:
            The number of repetitions.
        """
        return self._reps

    @reps.setter
    def reps(self, repetitions: int) -> None:
        """Set the repetitions.

        Args:
            repetitions: The new repetitions.

        Raises:
            ValueError: If reps setter has parameter repetitions < 1.
        """
        if repetitions < 1:
            raise ValueError("The repetitions should be larger than or equal to 1")
        if repetitions != self._reps:
            self._invalidate()
            self._reps = repetitions

    def print_settings(self) -> str:
        """Returns information about the setting.

        Returns:
            The class name and the attributes/parameters of the instance as ``str``.
        """
        ret = f"Qcnn: {self.__class__.__name__}\n"
        params = ""
        for key, value in self.__dict__.items():
            if key[0] == "_":
                params += f"-- {key[1:]}: {value}\n"
        ret += f"{params}"
        return ret

    def _check_configuration(self, raise_on_failure: bool = True) -> bool:
        """Check if the configuration of the class is valid.

        Args:
            raise_on_failure: Whether to raise on failure.

        Returns:
            True, if the configuration is valid and the circuit can be constructed. Otherwise
            an ValueError is raised.

        Raises:
            ValueError: If the number of repetitions is not set.
            ValueError: If the number of qubits is not set.
        """
        valid = True
        if self.num_qubits is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of qubits specified.")

        if self.reps is None:
            valid = False
            if raise_on_failure:
                raise ValueError("No number of repetitions specified.")

        return valid

    @property
    def initial_state(self) -> QuantumCircuit:
        """Return the initial state that is added in front of the circuit.

        Returns:
            The initial state.
        """
        return self._initial_state

    @initial_state.setter
    def initial_state(self, initial_state: QuantumCircuit) -> None:
        """Set the initial state.

        Args:
            initial_state: The new initial state.

        Raises:
            ValueError: If the number of qubits has been set before and the initial state
                does not match the number of qubits.
        """
        self._initial_state = initial_state
        self._invalidate()

    def assign_parameters(
        self,
        parameters: Mapping[Parameter, ParameterExpression | float]
        | Sequence[ParameterExpression | float],
        inplace: bool = False,
    ) -> QuantumCircuit | None:
        """Assign parameters to the circuit.

        This method also supports passing a list instead of a dictionary. If a list
        is passed, the list must have the same length as the number of unbound parameters in
        the circuit.

        Returns:
            A copy of the circuit with the specified parameters.

        Raises:
            AttributeError: If the parameters are given as list and do not match the number
                of parameters.
        """
        if parameters is None or len(parameters) == 0:
            return self

        if not self._is_built:
            self._build()

        return super().assign_parameters(parameters, inplace=inplace)

    @property
    def num_parameters_settable(self) -> int:
        """The number of total parameters that can be set to distinct values.

        This does not change when the parameters are bound or exchanged for same parameters,
        and therefore is different from ``num_parameters`` which counts the number of unique
        :class:`~qiskit.circuit.Parameter` objects currently in the circuit.

        Returns:
            The number of parameters originally available in the circuit.

        Note:
            This quantity does not require the circuit to be built yet.
        """
        num = 0

        total_num_qubits = self.num_qubits
        while total_num_qubits > 1:
            block_num_qubits = int(numpy.ceil(total_num_qubits / 2))

            # Convolutional Layer
            if total_num_qubits > 2:
                num += total_num_qubits * self._conv_template_num_parameters
            else:
                num += self._conv_template_num_parameters

            # Pooling Layer
            num += total_num_qubits // 2 * self._pool_template_num_parameters

            total_num_qubits = block_num_qubits

        return num

    def _conv_layer(self, num_qubits, params):
        qc = QuantumCircuit(num_qubits, name="Convolutional")
        qubits = list(range(num_qubits))
        param_index = 0
        for q1, q2 in zip(qubits[0::2], qubits[1::2]):
            qc = qc.compose(
                self._conv_template(
                    params[param_index : (param_index + self._conv_template_num_parameters)]
                ),
                [q1, q2]
            )
            if self._insert_barriers:
                qc.barrier()
            param_index += self._conv_template_num_parameters

        if num_qubits > 2:
            for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
                qc = qc.compose(
                    self._conv_template(
                        params[param_index : (param_index + self._conv_template_num_parameters)]
                    ),
                    [q1, q2]
                )
                if self._insert_barriers:
                    qc.barrier()
                param_index += self._conv_template_num_parameters

        return qc.to_instruction()

    def _pool_layer(self, sources, sinks, params):
        num_qubits = len(sources) + len(sinks)
        qc = QuantumCircuit(num_qubits, name="Pooling")
        param_index = 0
        for source, sink in zip(sources, sinks):
            qc = qc.compose(
                self._pool_template(
                    params[param_index : (param_index + self._pool_template_num_parameters)]
                ),
                [source, sink]
            )
            if self._insert_barriers:
                qc.barrier()
            param_index += self._pool_template_num_parameters

        return qc.to_instruction()

    def _build(self) -> None:
        """If not already built, build the circuit."""
        if self._is_built:
            return

        super()._build()

        if self.num_qubits == 0:
            return

        circuit = QuantumCircuit(*self.qregs, name=self.name)
        params = ParameterVector(self._parameter_prefix, length=self.num_parameters_settable)

        # use the initial state as starting circuit, if it is set
        if self.initial_state:
            circuit.compose(self.initial_state.copy(), inplace=True)
            if self._insert_barriers:
                circuit.barrier()

        param_index = 0
        total_num_qubits = self.num_qubits
        while total_num_qubits > 1:
            block_num_qubits = int(numpy.ceil(total_num_qubits / 2))

            # Convolutional Layer
            circuit.compose(
                self._conv_layer(
                    total_num_qubits,
                    params[param_index :]
                ),
                list(range(self.num_qubits - total_num_qubits, self.num_qubits)),
                inplace=True
            )
            if total_num_qubits > 2:
                param_index += total_num_qubits * self._conv_template_num_parameters
            else:
                param_index += self._conv_template_num_parameters

            # Pooling Layer
            circuit.compose(
                self._pool_layer(
                    list(range(block_num_qubits)),
                    list(range(block_num_qubits, total_num_qubits)),
                    params[param_index :]
                ),
                list(range(self.num_qubits - total_num_qubits, self.num_qubits)),
                inplace=True
            )
            param_index += total_num_qubits // 2 * self._pool_template_num_parameters

            total_num_qubits = block_num_qubits

        for _ in range(self._reps):
            self.append(circuit, self.qubits)
