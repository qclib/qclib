import numpy as np

def pqm(circuit, pattern, q_memory, q_auxiliary, is_classical_pattern=False):
    r"""
    Prepares a circuit which the output is determined by a probability distribution on the memory
    which is peaked around the stored patterns closest in Hamming distance to the input.
    Developed by `C. A. Trugenberger (2001) <https://arxiv.org/pdf/quant-ph/0012100v2.pdf>`_.
    The retrieval algorithm requires three registers. The first, of ``n`` qubits, contains the input
    pattern; the second, also of ``n`` qbits, contains the memory; and finally there is a single qubit
    auxiliary register.
    .. note::
        The operator ``U``, used in Trugenberger's article, has been replaced by ``XPX`` (where ``P`` is the phase
        gate). Therefore, we removed the ``X`` (``NOT``) operators from equations (12) and (16) of the 
        paper.
    .. note::
        If the pattern is classical (``is_classical_pattern = True``), the ``CNOT`` operators of equations
        (12) and (16) are applied directly to the memory qubits using ``NOT`` gates (the article
        names the ``CNOT`` as ``XOR``). Otherwise, ``CNOT's`` controlled by the pattern qubits are used and 
        applied to the memory qubits. In the latter case, ``n`` additional qubits are needed, compared 
        to the classical pattern case.
    Args:
        circuit: a qiskit quantum circuit.
        pattern: a list of bits (0 and 1 ints) or a basis encoded quantum register .
        q_memory: an amplitude encoded quantum register with memory data (superposition of ``p`` patterns on ``n`` entangled qbits).
        q_auxiliary: an uninitialized one-qubit quantum register.
        is_classical_pattern: indicates if ``pattern`` is a classical data (list of bits) or a quantum register.
    """
    size = len(q_memory)

    circuit.h(q_auxiliary)
        
    if (is_classical_pattern):          
        for k, q_m in enumerate(q_memory): # classical pattern register
            if (pattern[k]==1):            
                circuit.x(q_m)
    else:
        for k, q_m in enumerate(q_memory): # quantum pattern register
            circuit.cx(pattern[k], q_m)    

    for k, q_m in enumerate(q_memory):
        circuit.p(-np.pi / (2 * size), q_m)
    
    for k, q_m in enumerate(q_memory):
        circuit.cp( np.pi / size, q_auxiliary, q_m)
    
    if (is_classical_pattern):
        for k, q_m in list(enumerate(q_memory))[::-1]: # classical pattern register
            if (pattern[k]==1):            
                circuit.x(q_m)
    else:
        for k, q_m in list(enumerate(q_memory))[::-1]: # quantum pattern register
            circuit.cx(pattern[k], q_m)  

    circuit.h(q_auxiliary)
