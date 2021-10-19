"""
n-qubit toffoli gate
"""
from typing import NamedTuple
import qiskit
import numpy as np


def toffoli(gate:np.ndarray, qcirc: qiskit.QuantumCircuit, controls: list, targ: int):
    """

    Parameters
    ----------
    qcirc: qiskit.QuantumCircuit
    controls: list of control qubits
    targ: target qubit

    Returns
    -------

    """

    n_qubits = len(controls) + 1
    gate_circuit = qiskit.QuantumCircuit(n_qubits, name="T" + str(targ))
    coef = _coefficients(n_qubits)

    c1c2(gate, coef, n_qubits, gate_circuit)
    _c3(gate, coef, n_qubits, gate_circuit)

    c1c2(gate, coef[:-1,1:-1], n_qubits-1, gate_circuit, False)
    _c3(gate, coef[:-1, 1:-1], n_qubits - 1, gate_circuit, False)

    qcirc.compose(gate_circuit, controls + [targ], inplace=True)


def _c3(gate, coef, n_qubits, qcirc, first=True):
    """ phase 2 """
    for k in range(1, n_qubits-1):
        line = k
        column = n_qubits - 1

        while column < 2*n_qubits-3 and coef[line, column] != 0:
            control = (column % (n_qubits - 1) + 1)
            targ = line + 1
            # -----
            if first and targ == n_qubits-1:
                csqgate = _xgate(gate, coef, column, line)
                qcirc.compose(csqgate, qubits=[control, targ], inplace=True)
            else:
                qcirc.crx(np.pi / coef[line, column], control, targ)

            line = line - 1
            column = column + 1

    for k in range(n_qubits, 2*n_qubits-3):
        line = n_qubits-2
        column = k
        control = column - line
        while column< 2*n_qubits-3 and coef[line, column] != 0:
            targ = line + 1
            # -----
            if first and targ == n_qubits - 1:
                csqgate = _xgate(gate, coef, column, line)
                qcirc.compose(csqgate, qubits=[control, targ], inplace=True)
            else:
                qcirc.crx(np.pi / coef[line, column], control, targ)

            line = line - 1
            column = column + 1
            control = control + 1


def c1c2(gate, coef, n_qubits, qcirc, first=True):
    """ phase 1 """
    if not first:
        for k in range(n_qubits-1):
            coef[k, n_qubits-2] = - coef[k, n_qubits-2]

    for k in range(n_qubits - 1):
        line = n_qubits - 2
        column = k
        while line >= 0 and column >= 0 and coef[line, column] != 0:
            control = n_qubits - 2 - column
            targ = line + 1
            # -----
            if first and targ == n_qubits - 1:
                csqgate = _xgate(gate, coef, column, line)
                qcirc.compose(csqgate, qubits=[control, targ], inplace=True)
            else:
                qcirc.crx(np.pi / coef[line, column], control, targ)

            line = line - 1
            column = column - 1



    for k in range(n_qubits - 3, -1, -1):
        column = n_qubits - 2
        line = k


        while line >= 0 and column >= 0 and coef[line, column] != 0:
            # -----
            control = n_qubits - 2 - column
            targ = line + 1
            if first and targ == n_qubits - 1:
                csqgate = _xgate(gate, coef, column, line)
                qcirc.compose(csqgate, qubits=[control, targ], inplace=True)
            else:
                qcirc.crx(np.pi / coef[line, column], control, targ)

            line = line - 1
            column = column - 1


def _xgate(agate, coef, column, line):
    signal = np.sign(coef[line, column])
    param = np.abs(coef[line, column])

    values, vectors = np.linalg.eig(agate)
    gate = np.power(values[0], 1/param) * vectors[:,[0]] @ vectors[:,[0]].conj().T + \
           np.power(values[1], 1/param) * vectors[:,[1]] @ vectors[:,[1]].conj().T

    if signal < 0:
        gate = np.linalg.inv(gate)

    sqgate = qiskit.QuantumCircuit(1, name='X^1/' + str(coef[line, column]))
    sqgate.unitary(gate, 0) #pylint: disable=maybe-no-member
    csqgate = sqgate.control(1)

    return csqgate


def _coefficients(n_qubits):
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

#
# if __name__ == '__main__':
#     from qclib.util import get_counts, get_state
#
#
#     qcirc = qiskit.QuantumCircuit(6)
#     # qcirc.x(0)
#     qcirc.x(1)
#     qcirc.x(2)
#     # qcirc.x(3)
#     # qcirc.x(4)
#
#
#     toffoli(qcirc, [0, 1, 2, 3, 4], 5)
#     qcirc.measure_all()
#     print(get_counts(qcirc))
#     print(1)
