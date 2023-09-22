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

from qclib.gates.initialize import Initialize
from qclib.gates.initialize_sparse import InitializeSparse


class InitializeAdaptive(Initialize):
    """
    Superclass dedicated for state preparation algorithms
    for initializing adaptative states.
    """

    def _get_num_qubits(self, params):
        if isinstance(params, dict):
            InitializeSparse._get_num_qubits(self, params)
        else:
            Initialize._get_num_qubits(self, params)

    def validate_parameter(self, parameter):
        if isinstance(parameter, tuple):
            InitializeSparse.validate_parameter(self, parameter)
        else:
            Initialize.validate_parameter(self, parameter)
