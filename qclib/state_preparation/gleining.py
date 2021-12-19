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

'''
Gleining et al algorithm for creating a quantum
circuit that loads a sparse state.
https://ieeexplore.ieee.org/document/9586240
'''

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import U3Gate
from random import randint

def initialize(state):
  state_dict = _build_state_dict(state)
  
  b_strings = list(state_dict.keys())
  
  n_qubits = len(b_strings[0])
  quantum_register =  QuantumRegister(n_qubits)
  quantum_circuit = QuantumCircuit(quantum_register)

  while len(b_strings) > 1:
    state_dict, quantum_circuit = _merging_procedure(state_dict, quantum_circuit)
    b_strings = list(state_dict.keys())

  x = b_strings.pop()
  for (bit_idx, bit)  in enumerate(x): 
    if bit == '1':
      quantum_circuit.x(bit_idx)

  return quantum_circuit.reverse_ops()



def _build_state_dict(state): 
  """
    Builds a dict of the non zero amplitudes with their
    associated binary strings as follows:
      { '000': <value>, ... , '111': <value> } 
    Args: 
      state: The classical description of the state vector
  """
  n_qubits = np.ceil(np.log(len(state))).astype(np.int)
  state_dict = {}
  for (value_idx, value) in enumerate(state):
    if value != 0: 
      binary_string = '{:0{}b}'.format(value_idx, n_qubits)
      state_dict[binary_string] = value
  return state_dict

def _maximizing_difference_bit_search(b_strings, dif_qubits):
  """
    Splits the set of bit strings into two (t_0 and t_1), by setting
    t_0 as the set of bit_strings with 0 in the b-th position, and 
    t_1 as the set of bit_strings with 1 in the b-th position. 
    Searching for the bit b that maximizes the difference between 
    t_0 and t_1 where neither is empty.
    Args: 
      b_string: A list of bit strings eg.: ['000', '011', ...,'101']
      dif_qubits: A list of previous qubits found to maximize the difference
    Returns:
      bit_index: The qubit index that maximizes the splitting of the list b_strings
      t_0: List of binary strings with 0 on the b-th qubit
      t_1: List of binary strings with 1 on the b-th qubit
  """
  t_0 = []
  t_1 = []
  bit_index = 0
  set_difference = 0
  bit_search_space = list(set(range(len(b_strings[0]))) - set(dif_qubits))
  for b in bit_search_space: 
    temp_t0 = []
    temp_t1 = []
    for bit_string in b_strings: 
      if bit_string[b] == '0': 
        temp_t0.append(bit_string)
      else: 
        temp_t1.append(bit_string)
    # Neither temp_t0 nor temp_t1 must be empty
    if (temp_t0 and temp_t1):
      temp_difference = np.abs(len(temp_t0) - len(temp_t1))
      if temp_difference == 0 and not t_0 and not t_1:
        t_0 = temp_t0
        t_1 = temp_t1
        bit_index = b
      elif temp_difference > set_difference:
        t_0 = temp_t0
        t_1 = temp_t1
        bit_index = b
        set_difference = temp_difference
  return bit_index, t_0, t_1

def _build_bit_string_set(b_strings, dif_qubits, dif_values):
  """
    Creates a new set of bit strings from b_strings, where the bits 
    in the indexes in dif_qubits match the values in dif_values. 
    
    Args: 
      b_strings: list of bit strings eg.: ['000', '011', ...,'101']
      dif_qubits: list of integers with the bit indexes
      dif_values: list of integers values containing the values each bit
                  with index in dif_qubits shoud have 
    Returns:
      A new list ot bit_strings, with matchin values in dif_values
      on indexes dif_qubits
  """
  bit_string_set = [] 
  
  for b_string in b_strings: 
    include_string = True
    for (b_index, b_value) in zip(dif_qubits, dif_values):
      if b_string[b_index] != b_value:
        include_string = False
    if include_string:
      bit_string_set.append(b_string)
  return bit_string_set

def _bit_string_search(b_strings, dif_qubits, dif_values):
  """
    Searches for the bit strings with unique qubit values in `dif_values` 
    on indexes `dif_qubits`. 
    Args: 
      b_strings: List of binary strings where the search is to be performed
                 e.g.: ['000', '010', '101', '111']
      dif_qubits: List of indices on a binary string of size N e.g.: [1, 3, 5]
      dif_values: List of values each qubit must have on indexes stored in dif_qubits [0, 1, 1]
    Returns: 
      b_strings: One size list with the string found, to have values dif_values on indexes
                 dif_qubits
      dif_qubits: Updated list with new indexes
      dif_values: Updated list with new values
  """
  temp_strings = b_strings
  while len(temp_strings) > 1: 
    b, t_0, t_1 = _maximizing_difference_bit_search(temp_strings, dif_qubits)
    dif_qubits.append(b)
    if len(t_0) <= len(t_1): 
      dif_values.append('0')
      temp_strings = t_0
    else: 
      dif_values.append('1')
      temp_strings = t_1
    # dif_qubits must have at least two values stored in it
    if len(temp_strings) == 1 and len(dif_qubits) == 1: 
      temp_strings = b_strings
    
  return temp_strings, dif_qubits, dif_values

def _search_bit_strings_for_merging(state_dict):
  """
    Searches for the states described by the bit strings x1 and x2 to be merged 
    Args: 
      state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                  binary strings as keys e.g.: {'001': <value>, '101': <value>}
    Returns: 
      x1: First binary string
      x2: Second binary string
      dif: Qubit index to be used as target for the merging operation
      dif_qubits: List of qubit indexes where x1 and x2 must be equal, because the correspondig qubits
                  of those indexes are to be used as control for the merging operation
  """
  # Initialization
  dif_qubits = []
  dif_values = []
  t = s = list(state_dict.keys())
  x1 = None
  x2 = None

  if len(t) == 2:
    # Search for the difference bit
    b, t_0, t_1 = _maximizing_difference_bit_search(t, dif_qubits)
    dif = b
    x1 = t_0[0]
    x2 = t_1[0]
  else:
    # Searching for x1
    t, dif_qubits, dif_values = _bit_string_search(t, dif_qubits, dif_values)
    dif = dif_qubits.pop()
    dif_values.pop()
    x1 = t[0]

    # Searching for x2
    t = _build_bit_string_set(s, dif_qubits, dif_values)
    t, dif_qubits, dif_values = _bit_string_search(t, dif_qubits, dif_values)
    x2 = t[0]

  return x1, x2, dif, dif_qubits

def _apply_operation_to_bit_string(b_string, operation, qubit_indexes):
  """
    Applies changes on binary strings according to the version
    Args:
      b_string: Binary string '00110'
      operation: Operation to be applied to the string
      qubit_indexes: Indexes of the qubits on the binary strings where the operations are to
                      be applied
    Returns: 
      Updated binary string
  """
  assert operation in ['x', 'cx']
  compute_op = None 
  if operation == 'x':
    compute_op = lambda x, idx : x[:idx] + '1' + x[idx+1:] if x[idx] == '0' else x[0:idx] + '0' + x[idx+1:]
  elif operation == 'cx': 
    compute_op = lambda x, idx: x[:idx[1]] + "{}".format((not int(x[idx[1]])) * 1) + x[idx[1]+1:] if x[idx[0]] == '1' else x
  return compute_op(b_string, qubit_indexes)

def _update_state_dict_according_to_operation(state_dict, operation, qubit_indexes, merge_strings=None):
  """
    Updates the keys of the state_dict according to the operation being applied to the circuit
    Args: 
      state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                  binary strings as keys e.g.: {'001': <value>, '101': <value>}
      operation: Operation to be applied to the states, it must be ['x', 'cx', 'merge']
      qubit_indexes: Indexes of the qubits on the binary strings where the operations are to
                      be applied
      merge_strings: Binary strings associated ot the states on the quantum processor
                     to be merge e.g.:['01001', '10110'] 
    Returns:
      A state_dict with the updated states
  """
  assert operation in ['x', 'cx', 'merge']
  state_list = list(state_dict.items())
  new_state_dict = {}
  if operation == 'merge': 
    assert merge_strings != None
    # Computes the norm of x1 and x2
    new_state_dict = state_dict.copy()
    norm = np.linalg.norm([new_state_dict[merge_strings[0]], new_state_dict[merge_strings[1]]])
    new_state_dict.pop(merge_strings[0], None)
    new_state_dict[merge_strings[1]] = norm
  else:
    for (bit_string, value) in state_list:
      temp_bstring = _apply_operation_to_bit_string(bit_string, operation, qubit_indexes)
      new_state_dict[temp_bstring] = value

  return new_state_dict    

def _equalize_bit_string_states(x1, x2, dif, state_dict, quantum_circuit):
  """
    Makes states represented by bit strings x1 and x2 equal at every qubit except at the one in the
    dif index. And alters the bit strings and state_dict accordingly.
    Args: 
      x1: Frist bit string
      x2: Second bit string
      dif: index where both x1 and x2 mus be different
      state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                  binary strings as keys e.g.: {'001': <value>, '101': <value>}
      quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit
    Returns: 
      Updated x1, x2, state_dict and quantum_circuit
  """
  b_index_list = list(range(len(x1)))
  b_index_list.remove(dif)

  for b_index in b_index_list:
    if x1[b_index] != x2[b_index]:
      quantum_circuit.cx(dif, b_index)
      x1 = _apply_operation_to_bit_string(x1, 'cx', [dif, b_index])
      x2 = _apply_operation_to_bit_string(x2, 'cx', [dif, b_index])
      state_dict = _update_state_dict_according_to_operation(state_dict, 'cx', [dif, b_index])
  
  return x1, x2, state_dict, quantum_circuit

def _apply_not_gates_to_qubit_index_list(x1, x2, dif_qubits, state_dict, quantum_circuit):
  """
    Makes states represented by bit strings x1 and x2 equal at every qubit except at the one in the
    dif index. And alters the bit strings and state_dict accordingly.
    Args: 
      x1: Frist bit string
      x2: Second bit string
      dif_qubits: indexes where both x1 and x2 are equal
      state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                  binary strings as keys e.g.: {'001': <value>, '101': <value>}
      quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit
    Returns: 
      Updated x1, x2, state_dict and quantum_circuit
  """
  for b_index in dif_qubits: 
    if x2[b_index] != '1':
      quantum_circuit.x(b_index)
      x1 = _apply_operation_to_bit_string(x1, 'x', b_index)
      x2 = _apply_operation_to_bit_string(x2, 'x', b_index)
      state_dict = _update_state_dict_according_to_operation(state_dict, 'x', b_index)
  return x1, x2, state_dict, quantum_circuit

def _preprocess_states_for_merging(x1, x2, dif, dif_qubits, state_dict, quantum_circuit):
  """
    Apply the operations on the basis states to prepare for merging x1 and x2. 
    Args: 
      state_dict: A dictionary with the non-zero amplitudes associated to their corresponding
                  binary strings as keys e.g.: {'001': <value>, '101': <value>}
      x1: First binary string to be merged
      x2: Second binary string to be merged
      dif_qubits: List of qubit indexes on the binary strings 
      dif: Target qubit index where the merge operation is to be applied
      quantum_circuit: Qiskit's QuantumCircuit object where the operations are to be called
    Returns: 
      state_dict: Updated state dict
      x1: First updated binary string to be merge
      x2: Second updated binary string to be merged
      quantum_circuit: Qiskit's quantum circuit's object with the gates applied to the circuit 
  """
  
  if x1[dif] != '1': 
    quantum_circuit.x(dif)
    x1 = _apply_operation_to_bit_string(x1, 'x', dif)
    x2 = _apply_operation_to_bit_string(x2, 'x', dif)
    state_dict = _update_state_dict_according_to_operation(state_dict, 'x', dif)

  x1, x2, state_dict, quantum_circuit = _equalize_bit_string_states(x1, x2, dif, state_dict, quantum_circuit)

  x1, x2, state_dict, quantum_circuit = _apply_not_gates_to_qubit_index_list(x1, x2, dif_qubits, state_dict, quantum_circuit)
  
  return x1, x2, state_dict, quantum_circuit

def _compute_angles(amplitude_1, amplitude_2):
  """
    Computes the angles for the adjoint of the merge matrix M
    that is going to map the dif qubit to zero e.g.: 
      M(a|0> + b|1>) -> |0>

    Args: 
      amplitude_1: A complex/real value, associated with the string with
                   1 on the dif qubit
      amplitude_2: A complex/real value, associated with the string with
                   0 on the dif qubit
    Returns: 
      The angles theta, lambda and phi for the U3 operator
  """
  norm = np.linalg.norm([amplitude_1, amplitude_2])
  amplitude_1 = np.conj(amplitude_1 / norm)
  amplitude_2 = np.conj(amplitude_2 / norm)
  
  phi = 0
  lamb = 0 
  # there is no minus on the theta because the intetion is to compute the inverse
  if isinstance(amplitude_1, complex) and isinstance(amplitude_2, complex): 
    theta = 2 * np.arcsin(amplitude_2)
    lamb = np.log(amplitude_2).imag
    phi = np.log(amplitude_1).imag - lamb
  else: 
    theta = 2 * np.arcsin(amplitude_2)

  return theta, lamb, phi

def _merging_procedure(state_dict, quantum_circuit):

  x1, x2, dif, dif_qubits = _search_bit_strings_for_merging(state_dict)

  # Cricuit building 
  x1, x2, state_dict, quantum_circuit = _preprocess_states_for_merging(x1, 
                                                                       x2, 
                                                                       dif,
                                                                       dif_qubits, 
                                                                       state_dict,
                                                                       quantum_circuit)

  theta, phi, lamb = _compute_angles(state_dict[x1], state_dict[x2])
  
  # Applying merge operation
  merge_gate = None
  if not dif_qubits:
    merge_gate = U3Gate(theta, phi, lamb, label='U3')
  else:
    merge_gate = U3Gate(theta, phi, lamb, label='U3').control(num_ctrl_qubits=len(dif_qubits))
  quantum_circuit.append(merge_gate, dif_qubits+[dif], [])
  state_dict = _update_state_dict_according_to_operation(state_dict, 
                                                         'merge',
                                                         None,
                                                         merge_strings=[x1, x2] )
  return state_dict, quantum_circuit
