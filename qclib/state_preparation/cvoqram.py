from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import U3Gate
from qclib.util import _compute_matrix_angles
import numpy as np


class CVOQRAM:
    def __init__(self, nbits, data):

        self.initialization(nbits)
        self.circuit.x(self.u[0])
        for k, binary_string_end_feature in enumerate(data):
            binary_string, feature = binary_string_end_feature 
    
            self.control = CVOQRAM.select_controls(binary_string)
            self.flip_flop()
            self._load_superposition(feature)
            if k<len(data)-1:
                self.flip_flop()
            else:
                break              


    def initialization(self, nbits):      
        self.u = QuantumRegister(1, name='u')
        self.memory = QuantumRegister(nbits, name='m')
        self.anc    = QuantumRegister(nbits-1, name='anc')
        self.circuit = QuantumCircuit(self.u, self.anc, self.memory)
        self.nbits = nbits
        self.norm = 1


    def flip_flop(self):
        for k in self.control:
            self.circuit.cx(self.u[0], self.memory[k])    

    @staticmethod
    def select_controls(binary_string):
        control = []
        for k, bit in enumerate(binary_string):
            if bit == '1':
                control.append(k)
        return control



    def mcuvchain(self, alpha, beta, phi):
        """
         N-qubit controlled-unitary gate
        """        
        
        
        lst_ctrl = self.control
        lst_ctrl_reversed = list(reversed(lst_ctrl))        
        self.circuit.rccx(self.memory[lst_ctrl_reversed [0]],
                          self.memory[lst_ctrl_reversed[1]], 
                          self.anc[self.nbits-2])       
        
        tof = {}
        i = self.nbits-1        
        for ctrl in lst_ctrl_reversed [2:]:    
            self.circuit.rccx(self.anc[i-1], 
                              self.memory[ctrl], 
                              self.anc[i-2])
            tof[ctrl] = [i-1, i-2]
            i-=1
        #self.ugate_control(self.anc[i-1],self.u[0], U, 'V')
        self.circuit.cu3(alpha, beta, phi, self.anc[i-1], self.u[0])

        for ctrl in lst_ctrl[:-2]:
            self.circuit.rccx(self.anc[tof[ctrl][0]],
                              self.memory[ctrl], 
                              self.anc[tof[ctrl][1]])
            
        self.circuit.rccx(self.memory[lst_ctrl[-1]],
                          self.memory[lst_ctrl[-2]], 
                          self.anc[self.nbits-2])   



    def _load_superposition(self, feature):
        """
        Load pattern in superposition
        """

        alpha, beta, phi = _compute_matrix_angles(feature, self.norm)
        U = U3Gate(alpha, beta, phi)       
        
        if len(self.control) == 0:            
             self.circuit.u(alpha, beta, phi, self.u[0])       
        elif len(self.control) == 1:
            self.circuit.cu3(alpha, beta, phi, self.memory[self.control[0]], self.u[0])
        else:            
            self.mcuvchain(alpha, beta, phi)
        self.norm = self.norm - np.absolute(np.power(feature, 2))
       
def cvoqram_initialize(state):
    """
    Creates a circuit to initialize a quantum state arXiv:

    For instance, to initialize the state a|001>+b|100>
        $ state = [('001', a), ('100', b)]
        $ circuit = sparse_initialize(state)

    Parameters
    ----------
    state: list of [(str,float)]
        A unit vector representing a quantum state.
        str: binary string
        float: amplitude

    Returns
    -------
    QuantumCircuit to initialize the state

    """
    qbit = state[0][0]
    size = len(qbit)
    n_qubits = int(size)
    memory = CVOQRAM(n_qubits, state)
    return memory.circuit