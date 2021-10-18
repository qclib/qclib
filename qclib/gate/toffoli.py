"""
n-qubit toffoli gate
"""
from typing import NamedTuple
import qiskit
import numpy as np


def toffoli(qcirc: qiskit.QuantumCircuit, controls: list, targ: int):
    """

    Parameters
    ----------
    qcirc: qiskit.QuantumCircuit
    controls: list of control qubits
    targ: target qubit
    first: True

    Returns
    -------

    """

    n_qubits = len(controls) + 1
    gate_circuit = qiskit.QuantumCircuit(n_qubits, name="T" + str(targ))
    coef = _coefficients(n_qubits)

    c1c2(coef, n_qubits, gate_circuit)
    c3(coef, n_qubits, gate_circuit)

    c1c2(coef[:-1,1:-1], n_qubits-1, gate_circuit, False)
    c3(coef[:-1,1:-1], n_qubits-1, gate_circuit)

    qcirc.compose(gate_circuit, controls + [targ], inplace=True)


def c3(coef, n_qubits, qcirc):
    for k in range(n_qubits - 2, 1, -1):
        line = k
        column = n_qubits - 1
        while line <= n_qubits - 2:
            qcirc.crx(np.pi / coef[line, column], (column % (n_qubits - 1) + 1), line + 1)
            line = line + 1
            column = column + 1
    for k in range(1, n_qubits - 1):
        qcirc.crx(-np.pi / 2, k, k + 1)


def c1c2(coef, n_qubits, qcirc, first=True):
    if not first:
        for k in range(n_qubits-1):
            coef[k, n_qubits-2] = - coef[k, n_qubits-2]

    for k in range(n_qubits - 1):
        line = n_qubits - 2
        column = k
        while line >= 0 and column >= 0 and coef[line, column] != 0:
            control = n_qubits - 2 - column
            targ = line + 1

            if first and line == n_qubits-2:
                csqgate = xgate(coef, column, line)
                qcirc.compose(csqgate, qubits=[control, targ], inplace=True)

            else:
                qcirc.crx(np.pi / coef[line, column], control, targ)

            line = line - 1
            column = column - 1



    for k in range(n_qubits - 3, -1, -1):
        column = n_qubits - 2
        line = k
        while line >= 0 and column >= 0 and coef[line, column] != 0:
            qcirc.crx(np.pi / coef[line, column], n_qubits - 2 - column, line + 1)
            line = line - 1
            column = column - 1


def xgate(coef, column, line):
    param = np.abs( 1 / coef[line, column])
    plus = (1 / np.sqrt(2)) * np.array([[1], [1]])
    minus = (1 / np.sqrt(2)) * np.array([[1], [-1]])

    gate = np.power(1 + 0j, 1 / param) * plus @ plus.T + \
           np.power(-1 + 0j, 1 / param) * minus @ minus.T

    sqgate = qiskit.QuantumCircuit(1, name='X^1/' + str(coef[line, column]))
    sqgate.unitary(gate, 0)
    csqgate = sqgate.control(1)

    return csqgate


class _Flags(NamedTuple):
    mid: bool
    inverse: bool
    first: bool


def _coluna(qcirc, targs, control, flags):
    if flags.mid:
        k = 0
    else:
        k = 1

    if flags.inverse:
        signal = -1
    else:
        signal = 1

    for target in targs[:-1]:
        qcirc.crx(np.pi / (signal * 2 ** k), control, target)
        k = k + 1

    plus = (1/np.sqrt(2)) * np.array([[1], [1]])
    minus = (1/np.sqrt(2)) * np.array([[1], [-1]])

    gate = np.power(1+0j, 1/(signal*2**k)) * plus @ plus.T +\
                np.power(-1+0j, 1/(2**k)) * minus @ minus.T

    sqgate = qiskit.QuantumCircuit(1, name='X^1/' + str(2**k))
    if signal == 1:
        sqgate.unitary(gate, 0) #pylint: disable=maybe-no-member
    else:
        sqgate.unitary(np.linalg.inv(gate), 0)  # pylint: disable=maybe-no-member
    csqgate = sqgate.control(1)
    csqgate.name = "name=X^(1/?)"

    if flags.first:
        qcirc.compose(csqgate, qubits=[control, targs[-1]], inplace=True)
    else:
        qcirc.crx(np.pi / (signal * 2 ** k), control, targs[-1])

    return qcirc




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

if __name__ == '__main__':
    qcirc = qiskit.QuantumCircuit(6)
    toffoli(qcirc, [5,4,3,2,1], 0)