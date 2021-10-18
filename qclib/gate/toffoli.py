"""
n-qubit toffoli gate
"""
from typing import NamedTuple
import qiskit
import numpy as np


def toffoli(qcirc: qiskit.QuantumCircuit, controls: list, targ: int, first=True):
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
    n_controls = len(controls)

    for k in range(n_controls-1):
        _coluna(qcirc, controls[n_controls-k:] + [targ],
                controls[-1-k],
                _Flags(False, False, first))

    _coluna(qcirc, controls[1:] + [targ], controls[0], _Flags(True, not first, first))

    for k in range(n_controls-2, -1, -1):
        _coluna(qcirc, controls[n_controls-k:] + [targ], controls[-1-k], _Flags(False, True, first))

    if first:
        toffoli(qcirc, controls[:-1], controls[-1], first=False)


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
