import numpy as np
from dataclasses import dataclass
from qclib.state_preparation.bidirectional.tree_utils import *
from qclib.state_preparation.bidirectional.state_tree_preparation import *

@dataclass
class NodeAngleTree:
    """
    Binary tree node used in function create_angles_tree
    """
    index: int
    level: int
    angle_y: float
    angle_z: float
    left: 'NodeAngleTree'
    right: 'NodeAngleTree'
        
    def __str__(self):
        return str(self.level) + '_' + \
               str(self.index) + '\n' + \
               '{0:.2g}'.format(self.angle_y) + '_' + \
               '{0:.2g}'.format(self.angle_z) 

def create_angles_tree(state_tree):
    """
    :param state_tree: state_tree is an output of state_decomposition function
    :param tree: used in the recursive calls
    :return: tree with angles that will be used to perform the state preparation
    """
    angle_y = 0.0
    angle_z = 0.0
    if state_tree.right:
        amp = 0.0
        if state_tree.amplitude != 0.0:
            amp = state_tree.right.amplitude / state_tree.amplitude
        angle_y = 2 * np.arcsin( np.abs(amp) )
        angle_z = np.angle(amp)

    node = NodeAngleTree(state_tree.index, state_tree.level, angle_y, angle_z, None, None)

    if not is_leaf(state_tree.right):
        node.right = create_angles_tree(state_tree.right) 
        node.left = create_angles_tree(state_tree.left)
    
    return node

