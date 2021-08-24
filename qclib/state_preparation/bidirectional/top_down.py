import numpy as np
import qiskit

from qclib.state_preparation.bidirectional.state_tree_preparation import *
from qclib.state_preparation.bidirectional.angle_tree_preparation import *
from qclib.state_preparation.bidirectional.tree_circuit_utils     import *

def top_down_tree_walk(angle_tree, circuit, start_level, control_nodes=None, target_nodes=None):

    if angle_tree:
        if angle_tree.level < start_level:
            top_down_tree_walk(angle_tree.left, circuit, start_level)
            top_down_tree_walk(angle_tree.right, circuit, start_level)
        else:
            if target_nodes == None:
                control_nodes = []                           # initialize the controls
                target_nodes = [angle_tree]                  # start by the subtree root
            else:
                target_nodes = children(target_nodes)        # all the nodes in the current level

            angles_y = [node.angle_y for node in target_nodes]
            angles_z = [node.angle_z for node in target_nodes]
            target_qubit = target_nodes[0].qubit
            control_qubits = [node.qubit for node in control_nodes]
            circuit.ucry(angles_y, control_qubits[::-1], target_qubit)     # qiskit reverse
            if (any(angles_z) != 0.0):            
                circuit.ucrz(angles_z, control_qubits[::-1], target_qubit) # qiskit reverse
            
            control_nodes.append(angle_tree)                 # add current node to the controls list

                                                             # walk to the first node of the next level.
            top_down_tree_walk(angle_tree.left, circuit, start_level, control_nodes=control_nodes, target_nodes=target_nodes)

def initialize(circuit, state):

    n_qubits = int(np.log2(len(state)))

    state_tree = state_decomposition(n_qubits, state)
    angle_tree = create_angles_tree(state_tree)

    add_register(circuit, angle_tree, 0)

    top_down_tree_walk(angle_tree, circuit, 0)
    
    q_output = []
    output(angle_tree, q_output)
    
    return q_output, state_tree, angle_tree

