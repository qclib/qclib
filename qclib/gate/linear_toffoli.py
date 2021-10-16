"""
Linear depth 'toffoli' gate
"""
import qiskit
import numpy as np



def toffoli(qcirc, controls, targ, first=True):
    n_controls = len(controls)

    for k in range(n_controls-1):
        _coluna(qcirc, controls[n_controls-k:] + [targ], controls[-1-k], first=first)

    _coluna(qcirc, controls[1:] + [targ], controls[0], mid=True, inverse=not first, first=first)

    for k in range(n_controls-2, -1, -1):
        _coluna(qcirc, controls[n_controls-k:] + [targ], controls[-1-k], inverse=True, first=first)

    if first:
        toffoli(qcirc, controls[:-1], controls[-1], first=False)




def _coluna(qcirc: qiskit.QuantumCircuit, targs: list, control: float, mid=False, inverse=False, first=True):
    if mid:
        k = 0
    else:
        k = 1

    if inverse:
        signal = -1
    else:
        signal = 1

    for target in targs[:-1]:
        qcirc.crx(np.pi / (signal * 2 ** k), control, target)
        k = k + 1

    plus = (1/np.sqrt(2)) * np.array([[1], [1]])
    minus = (1/np.sqrt(2)) * np.array([[1], [-1]])

    gate = np.power(1+0j, 1/(signal*2**k)) * plus @ plus.T + \
           np.power(-1+0j, 1/(2**k)) * minus @ minus.T

    sqgate = qiskit.QuantumCircuit(1, name='X^1/' +str(signal*2**k))
    sqgate.unitary(gate, 0)
    csqgate = sqgate.control(1)
    csqgate.name = "name=X^(1/?)"

    if first:
        qcirc.compose(csqgate, qubits=[control, targs[-1]], inplace=True)
    else:
        qcirc.crx(np.pi / (signal * 2 ** k), control, targs[-1])

    return qcirc

def coefficients(n_qubits):
    coef = np.zeros((n_qubits - 1, 2 * n_qubits - 3))
    for i in range(0, n_qubits - 2):
        one_coef = 2 ** (i+1)
        for j in range(n_qubits - 2, -1, -1):
            coef[j, i] = one_coef
            coef[j, (-i + 2 * n_qubits - 4)] = -one_coef
            one_coef = one_coef // 2
            if one_coef < 2:
                break
    for k in range(n_qubits - 1):
        coef[k][n_qubits - 2] = 2 ** k
    return coef

if __name__ == '__main__':

    a = coefficients(6)
    print(a)

    # nqubits = 5
    # # gates = c1c2(nqubits)
    # # print(gates)
    # circ = qiskit.QuantumCircuit(nqubits)
    # circ.x(0)
    # circ.x(1)
    #
    # # circ.mct(list(range(nqubits-1)), nqubits-1)
    # toffoli(circ, list(range(nqubits-1)), nqubits-1)
    # # print(circ.draw())
    # # phase 1
    # circ = _coluna(circ, [3], 2)
    # circ = _coluna(circ, [2, 3], 1)
    # circ = _coluna(circ, [1, 2, 3], 0, mid=True)
    # circ = _coluna(circ, [2, 3], 1, inverse=True)
    # circ = _coluna(circ, [3], 2, inverse=True)
    #
    # # phase 2
    # circ = _coluna(circ, [2], 1)
    # circ = _coluna(circ, [1, 2], 0, mid=True, inverse=True)
    # circ = _coluna(circ, [2], 1, inverse=True)
    #
    # print(get_state(circ))
    # circ.measure_all()
    # print('counts', get_counts(circ))
    # # print('depth', circ.depth())
    # #
    # qc = qiskit.transpile(circ, basis_gates=['u', 'cx'])
    # print('cx', qc.count_ops()['cx'])
    # print('circ depth', circ.depth())
    # print('qc depth', qc.depth())
    # dagqc = qiskit.converters.circuit_to_dag(qc)
    # print('dag depth', dagqc.depth())

    #
    # circ2 = qiskit.QuantumCircuit(4)
    # circ2.mct([0, 1, 2], 3)
    # qc2 = qiskit.transpile(circ2, basis_gates=['u', 'cx'])
    # print('depth', qc2.depth())
    # print(qc2.count_ops())
