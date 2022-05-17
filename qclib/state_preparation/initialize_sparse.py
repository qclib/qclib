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

from pytest import param
from qclib.state_preparation.initialize import Initialize

class InitializeSparse(Initialize): 

  def _get_num_qubits(self, params):
    bit_string = list(params.keys())[0]
    self.num_qubits = len(bit_string)

  def validate_parameter(self, parameter):
    if isinstance(parameter, tuple):
      if not match('(0|1)+', parameter[0]):
        raise Exception('Dictionary keys must be binary strings')
      return super().validate_parameter(parameter[1])
    else: 
      raise Exception(''.join('Input param must be a dictionary ',
                              'with pairs (binary_string, values)'))
