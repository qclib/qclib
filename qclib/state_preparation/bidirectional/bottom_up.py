import numpy as np
import qiskit

from qclib.state_preparation.bidirectional.state_tree_preparation import *
from qclib.state_preparation.bidirectional.angle_tree_preparation import *
from qclib.state_preparation.bidirectional.tree_circuit_utils     import *

def _apply_cswaps(angle_tree, circuit):

    if angle_tree.angle_y != 0.0:
        left = angle_tree.left
        right = angle_tree.right
        
        while left and right:
            circuit.cswap(angle_tree.qubit, left.qubit, right.qubit)
            
            left = left.left
            right = leftmost(right)

def bottom_up_tree_walk(angle_tree, circuit, start_level):

    if angle_tree and angle_tree.level < start_level:
    
        if (angle_tree.angle_y != 0.0):
            circuit.ry(angle_tree.angle_y, angle_tree.qubit)
            if (angle_tree.angle_z != 0.0):
                circuit.rz(angle_tree.angle_z, angle_tree.qubit)

        bottom_up_tree_walk(angle_tree.left, circuit, start_level)
        bottom_up_tree_walk(angle_tree.right, circuit, start_level)

        _apply_cswaps(angle_tree, circuit)
        
def initialize(circuit, state):

    n_qubits = int(np.log2(len(state)))

    state_tree = state_decomposition(n_qubits, state)
    angle_tree = create_angles_tree(state_tree)
    
    add_register(circuit, angle_tree, n_qubits)

    bottom_up_tree_walk(angle_tree, circuit, n_qubits)
    
    q_output = []
    output(angle_tree, q_output)
    
    return q_output, state_tree, angle_tree
