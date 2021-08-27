####### Parameters ########
import qiskit
import sys
sys.path.append('/home/lds/Documents/qclib/')
from qclib.cvoqram import CVOQRAM
import numpy as np
from qclib.state_preparation import sparse_initialize
from qclib.util import double_sparse
import qiskit
import matplotlib.pyplot as plt


nqubits = [ 4,5, 6, 7, 8, 9, 10]
from statistics import mean
fig, ax = plt.subplots(figsize=[7,5])
marker = ['-.o', '-.>', '-.x', '-.*']
for s in [1,2,3,4]:     
    print('s =',s)
    lst_cnots = []
    lst_ones = []
    lst_cnots_dense = []
    for n in nqubits:
        data = []
        data_ones = []
        density = (2**s)/(2**n)
        for i in range(10):
            
            vector = double_sparse(n, density, s, 0.8, 0.2)     
                
            memory = CVOQRAM(n, vector)
            circ = memory.circuit
            transpiled_circ = qiskit.transpile(circ,                             
                                    basis_gates=['u', 'cx'])
            cx = transpiled_circ.count_ops()['cx']
            data.append(cx)
           
        lst_cnots.append(mean(data))
        

        
    ax.plot(nqubits, lst_cnots, marker[s-1], label='s={}'.format(s))
    

    ax.text(n,lst_cnots[-1]+5 , 's={}'.format(s),
            verticalalignment='bottom',
            horizontalalignment='center',
            color='black', fontsize=10)
    ax.set_title("CVO-QRAM")
ax.set(xlabel='Number of qubits', 
        ylabel='CNOT gates')
plt.savefig("sparse-s.png")
plt.show()    

        
        
        
    
   
    
