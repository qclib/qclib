import numpy as np

def pqm(circ, bin_input, memory, auxiliary):
    """
    circ: QuantumCircuit
    bin_input: binary list
    memory: Memory quantum register
    auxiliary: auxiliary quantum register
    """

    size = int(len(memory))

    for k in range(size):
        if bin_input[k] == 1:
            circ.x(memory[k])

    for k in range(size):
        circ.p(np.pi / 2 * size, memory[k])

    # initialize auxiliary quantum bit |c>
    circ.h(auxiliary[0])
    for k in range(size):
        circ.cp(- np.pi / size, auxiliary[0], memory[k])
    circ.h(auxiliary[0])