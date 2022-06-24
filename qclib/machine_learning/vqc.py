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
Changes Qiskit's VQC class (line 67) to accept BaaFeatureVector feature map.
from: if isinstance(self._feature_map, RawFeatureVector):
to  : if isinstance(self._feature_map, (RawFeatureVector, BaaFeatureVector)):
https://github.com/Qiskit/qiskit-aqua/blob/c1564af8792c6664670807614a378147fd04d28f/qiskit/aqua/algorithms/classifiers/vqc.py

The correct approach is to create a RawFeatureVector base class that implements
only the common routines. Other classes, such as BaaFeatureVector, would
inherit from this new class implementing at least the __init__ and _build
functions. The current RawFeatureVector class would have to be renamed.
"""

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.aqua.algorithms import VQC as qiskit_vqc
from qiskit.aqua.algorithms.classifiers.vqc import return_probabilities
# from qiskit.circuit.library import BlueprintCircuit
from qiskit_machine_learning.circuit.library import RawFeatureVector
from .baa_feature_vector import BaaFeatureVector  # pylint: disable=relative-beyond-top-level

# pylint: disable=invalid-name


class VQC(qiskit_vqc):
    """
    Ideally, this modification should be made in the Qiskit class.
    See the comment at the beginning of this file.
    """

    def _get_prediction(self, data, theta):
        """Make prediction on data based on each theta.

        Args:
            data (numpy.ndarray): 2-D array, NxD, N data points, each with D dimension
            theta (list[numpy.ndarray]): list of 1-D array, parameters sets for variational form

        Returns:
            Union(numpy.ndarray or [numpy.ndarray], numpy.ndarray or [numpy.ndarray]):
                list of NxK array, list of Nx1 array
        """
        circuits = []

        num_theta_sets = len(theta) // self._var_form.num_parameters
        theta_sets = np.split(theta, num_theta_sets)

        def _build_parameterized_circuits():
            var_form_support = isinstance(self._var_form, QuantumCircuit) \
                or self._var_form.support_parameterized_circuit
            feat_map_support = isinstance(self._feature_map, QuantumCircuit) \
                or self._feature_map.support_parameterized_circuit

            # cannot transpile RawFeatureVector or BaaFeatureVector.
            # See the comment at the beginning of this file.
            if isinstance(self._feature_map, (RawFeatureVector, BaaFeatureVector)):
                feat_map_support = False

            if var_form_support and feat_map_support and self._parameterized_circuits is None:
                parameterized_circuits = self.construct_circuit(
                    self._feature_map_params, self._var_form_params,
                    measurement=not self._quantum_instance.is_statevector)
                self._parameterized_circuits = \
                    self._quantum_instance.transpile(parameterized_circuits)[0]

        _build_parameterized_circuits()
        for thet in theta_sets:
            for datum in data:
                if self._parameterized_circuits is not None:
                    curr_params = dict(zip(self._feature_map_params, datum))
                    curr_params.update(dict(zip(self._var_form_params, thet)))
                    circuit = self._parameterized_circuits.assign_parameters(curr_params)
                else:
                    circuit = self.construct_circuit(
                        datum, thet, measurement=not self._quantum_instance.is_statevector)
                circuits.append(circuit)

        results = self._quantum_instance.execute(
            circuits, had_transpiled=self._parameterized_circuits is not None)

        circuit_id = 0
        predicted_probs = []
        predicted_labels = []
        for _ in theta_sets:
            counts = []
            for _ in data:
                if self._quantum_instance.is_statevector:
                    temp = results.get_statevector(circuit_id)
                    outcome_vector = (temp * temp.conj()).real
                    # convert outcome_vector to outcome_dict, where key
                    # is a basis state and value is the count.
                    # Note: the count can be scaled linearly, i.e.,
                    # it does not have to be an integer.
                    outcome_dict = {}
                    bitstr_size = int(math.log2(len(outcome_vector)))
                    for i, _ in enumerate(outcome_vector):
                        bitstr_i = format(i, '0' + str(bitstr_size) + 'b')
                        outcome_dict[bitstr_i] = outcome_vector[i]
                else:
                    outcome_dict = results.get_counts(circuit_id)

                counts.append(outcome_dict)
                circuit_id += 1

            probs = return_probabilities(counts, self._num_classes)
            predicted_probs.append(probs)
            predicted_labels.append(np.argmax(probs, axis=1))

        if len(predicted_probs) == 1:
            predicted_probs = predicted_probs[0]
        if len(predicted_labels) == 1:
            predicted_labels = predicted_labels[0]

        return predicted_probs, predicted_labels
