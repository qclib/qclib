import numpy as np

def pqm(circuit, pattern, q_memory, q_auxiliary, is_classical_pattern=False):
    """
    circuit     : quantum circuit
    pattern     : binary input register (classical or quantum)
    q_memory    : memory quantum register
    q_auxiliary : auxiliary quantum register
    """
    
    size = len(q_memory)

    circuit.h(q_auxiliary)
        
    if (is_classical_pattern):          
        for k, q_m in enumerate(q_memory): # classical pattern register
            if (pattern[k]==0):            
                circuit.x(q_m)
    else:
        for k, q_m in enumerate(q_memory): # quantum pattern register
            circuit.cx(pattern[k], q_m)    
            circuit.x(q_m)

    for k, q_m in enumerate(q_memory):
        circuit.p(-np.pi / (2 * size), q_m)
    
    for k, q_m in enumerate(q_memory):
        circuit.cp( np.pi / size, q_auxiliary, q_m)
    
    if (is_classical_pattern):
        for k, q_m in list(enumerate(q_memory))[::-1]: # classical pattern register
            if (pattern[k]==0):            
                circuit.x(q_m)
    else:
        for k, q_m in list(enumerate(q_memory))[::-1]: # quantum pattern register
            circuit.x(q_m)                 
            circuit.cx(q_pattern[k], q_m)  

    circuit.h(q_auxiliary)
