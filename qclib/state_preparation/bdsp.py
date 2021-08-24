from qclib.state_preparation.bidirectional.state_tree_preparation import *
from qclib.state_preparation.bidirectional.angle_tree_preparation import *
from qclib.state_preparation.bidirectional.tree_circuit_utils     import *
from qclib.state_preparation.bidirectional.top_down               import top_down_tree_walk
from qclib.state_preparation.bidirectional.bottom_up              import bottom_up_tree_walk

def initialize(circuit, state, split=None):

    n_qubits = int(np.log2(len(state)))

    state_tree = state_decomposition(n_qubits, state)
    angle_tree = create_angles_tree(state_tree)
    
    if (split == None):
        split = int(ceil(n_qubits/2)) # sublinear

    add_register(circuit, angle_tree, n_qubits-split)

    top_down_tree_walk(angle_tree, circuit, n_qubits-split)
    bottom_up_tree_walk(angle_tree, circuit, n_qubits-split)
    
    q_output = []
    output(angle_tree, q_output)
    
    return q_output, state_tree, angle_tree

