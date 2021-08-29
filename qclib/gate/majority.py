import numpy as np
from itertools import combinations
   
def operate(circuit, controls, target):
    n = len(controls)
    log_n = int(np.floor( np.log2(n) ))
    
    n_min = int(np.ceil( n/2 ))
    n_max = 2**log_n
    
    n_controls = []
    n_controls.append(n_min)
    
    if (n_min != n_max):
        if (n_min % 2 != 0):
            n_controls.extend(range(n_min+1, n_max))
            
        n_controls.append(n_max)
        
    for r in n_controls:
        comb = combinations(controls, r)
        for c in comb:
            circuit.mcx([*c], target)
