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

from re import match
from qclib.state_preparation.initialize import Initialize


class InitializeSparse(Initialize):
    """
    Superclass dedicated for state preparation algorithms
    for initializing sparse states.
    """

    def _get_num_qubits(self, params):
        """
        Computes the number of qubits, based
        on the number of 0 or 1 characters in
        the dictionary key.
        """
        bit_string = list(params.keys())[0]
        self.num_qubits = len(bit_string)

    def validate_parameter(self, parameter):
        """
        Sparse preparation params are converted to a list of tuples when
        being validated, where each tuple contains the binary string
        and the value which represents the amplitude associated to the
        quantum state.
        """
        if isinstance(parameter, tuple):
            if not match('([01])+', parameter[0]):
                raise Exception('Dictionary keys must be binary strings')
            validated_value = super().validate_parameter(parameter[1])
            return parameter[0], validated_value

        raise Exception('Input param must be a dictionary with pairs (binary_string, values)')
