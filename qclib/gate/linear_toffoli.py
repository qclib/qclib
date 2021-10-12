"""
Linear depth 'toffoli' gate
"""
import qiskit
import numpy as np


def toffoli(qcirc, controls, targ, first=True):
    n_controls = len(controls)

    for k in range(n_controls-1):
        _coluna(qcirc, controls[n_controls-k:] + [targ], controls[-1-k])

    _coluna(qcirc, controls[1:] + [targ], controls[0], mid=True, inverse=not first)

    for k in range(n_controls-2, -1, -1):
        _coluna(qcirc, controls[n_controls-k:] + [targ], controls[-1-k], inverse=True)

    if first:
        toffoli(qcirc, controls[:-1], controls[-1], first=False)




def _coluna(qcirc: qiskit.QuantumCircuit, targs: list, control: float, mid=False, inverse=False):
    if mid:
        k = 0
    else:
        k = 1

    if inverse:
        signal = -1
    else:
        signal = 1

    for target in targs:
        qcirc.crx(np.pi / (signal * 2 ** k), control, target)
        k = k + 1

    return qcirc


if __name__ == '__main__':
    nqubits = 5
    circ = qiskit.QuantumCircuit(nqubits)

    # circ.mct(list(range(nqubits-1)), nqubits-1)
    toffoli(circ, list(range(nqubits-1)), nqubits-1)
    print(circ.draw())
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
    # print('depth', circ.depth())
    #
    qc = qiskit.transpile(circ, basis_gates=['u', 'cx'])
    print('cx', qc.count_ops()['cx'])
    print('circ depth', circ.depth())
    print('qc depth', qc.depth())
    dagqc = qiskit.converters.circuit_to_dag(qc)
    print('dag depth', dagqc.depth())

    #
    # circ2 = qiskit.QuantumCircuit(4)
    # circ2.mct([0, 1, 2], 3)
    # qc2 = qiskit.transpile(circ2, basis_gates=['u', 'cx'])
    # print('depth', qc2.depth())
    # print(qc2.count_ops())
