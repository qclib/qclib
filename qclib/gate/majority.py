import numpy as np
from itertools import combinations
   
def operate(circuit, controls, target):
    n = len(controls)
    n_min = int(np.ceil( np.log2(n/2) ))
    n_max = int(np.floor( np.log2(n) ))

    n_controls = []
    n_controls.append(n_min)
    if (n_min != n_max):
        n_controls.append(n_max)
    
    for r in n_controls:
        comb = combinations(controls, 2**r)
        for c in comb:
            circuit.mcx([*c], target)
