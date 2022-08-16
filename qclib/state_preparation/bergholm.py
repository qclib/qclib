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
      This module implements the state preparation proposed by
      Bergholm et al (2005) available in: 
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330
"""
import numpy as np
import cmath
from qiskit import QuantumCircuit, QuantumRegister

from qclib.state_preparation.initialize import Initialize
from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    state_decomposition,
)
from qclib.gates.uc_gate import UCGate
from qclib.state_preparation.util import tree_utils

class BergholmInitialize(Initialize):
    """
        https://journals.aps.org/pra/abstract/10.1103/PhysRevA.71.052330
    """

    def __init__(self, params, inverse=False, label=None):

        self._name = "bergholm"
        self._get_num_qubits(params)

        if label is None:
          self._label = "bergholm"

          if inverse:
                self._label = "bergholm_dg"

        super().__init__(self._name, self.num_qubits, params, label=self._label)

    def _define(self):
        self.definition = self._define_initialize()

    def _define_initialize(self):

        q_register = QuantumRegister(self.num_qubits)
        q_circuit = QuantumCircuit(q_register)
        data = [Amplitude(i, a) for i, a in enumerate(self.params)]
        tree_root = state_decomposition(self.num_qubits, data)
        
        tree_level = self.num_qubits
        
        parent_nodes = []
        children_nodes = [] 

        tree_utils.subtree_level_nodes(tree_root,
                                       tree_level - 1,
                                       parent_nodes)
        parent_amplitudes = self._get_amplitudes(parent_nodes)

        tree_utils.subtree_level_nodes(tree_root,
                                       tree_level,
                                       children_nodes)
        children_amplitudes = self._get_amplitudes(children_nodes)

        while tree_level > 0:
            #if tree_level != 0:
            #    parent_amplitudes = self._apply_diagonal(parent_amplitudes,
            #                                             children_amplitudes)

            current_level_mux = self._build_multiplexor(parent_amplitudes, children_amplitudes)        

            ucg = UCGate(current_level_mux, up_to_diagonal=True)
            
            controls = q_register[self.num_qubits - tree_level + 1:]
            target = [q_register[self.num_qubits - tree_level]]
            q_circuit.append(ucg, target + controls)

            #preparing for the next loop
            tree_level -= 1
            children_amplitudes = parent_amplitudes

            parent_nodes = []
            tree_utils.subtree_level_nodes(tree_root,
                                           tree_level - 1,
                                           parent_nodes)
            parent_amplitudes = self._get_amplitudes(parent_nodes)

        return q_circuit.inverse()

    def _apply_diagonal(self, parent_amplitudes, children_amplitudes):
        phases = np.angle(children_amplitudes)

        diag = np.exp(1j*phases)[::2]
        diag = np.multiply(diag, parent_amplitudes)
        return diag

    def _build_multiplexor(self, parent_amplitudes, children_amplitudes):
        """
        Infers the unitary to be used in the uniformily controlled multiplexor
        defined by Bergholm et al (2005). This procedure assumes that the right most
        child of a node is on a odd index.
        Args:
        parent_amplitudes: list of nodes from the previous level
        children_amplitudes: children of the paren nodes
        Returns:
        list of 2-by-2 numpy arrays with the desired unitary to be used
        in the multiplexor
        """
        gates = []
        len_pnodes = len(parent_amplitudes)
        len_snodes = len(children_amplitudes)
        for parent_idx, sibling_idx in zip(range(len_pnodes), range(0, len_snodes, 2)):
            amp_ket0 = (children_amplitudes[sibling_idx] / parent_amplitudes[parent_idx])
            amp_ket1 = (children_amplitudes[sibling_idx+1] / parent_amplitudes[parent_idx])
            gates += [self._get_branch_operator(amp_ket0, amp_ket1)]
        return gates

    def _get_branch_operator(self, amplitude_ket0, amplitude_ket1):
        """
        Returns the matrix operator that is going to split the qubit in to two components
        of a superposition
        Args:
        amplitude_ket0: Complex amplitude of the |0> component of the superpoisition
        amplitude_ket1: Complex amplitude of the |1> component of the superpoisition
        Returns: 
        A 2x2 numpy array defining the desired operator inferred from
        the amplitude argument
        """

        operator = np.array([[amplitude_ket0, 
                              -np.conj(amplitude_ket1)],
                             [amplitude_ket1, 
                              np.conj(amplitude_ket0)]])
        return np.conj(operator).T

    def _get_amplitudes(self, node_list): 
        """
        Extract amplitudes from node list and
        returns them as a numpy array
        """
        amps = []
        for node in node_list:
            amps.append(node.amplitude)
        return np.array(amps)

    @staticmethod
    def initialize(q_circuit, state, qubits=None):
      if qubits is None:
          q_circuit.append(BergholmInitialize(state).definition, q_circuit.qubits)
      else:
          q_circuit.append(BergholmInitialize(state).definition, qubits)