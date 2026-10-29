# Copyright 2021 qclib project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Interface to SynQ, a C++ quantum circuit synthesis library
(https://github.com/qsoftware/synq).

SynQ is an optional dependency. Install it with
``pip install git+https://github.com/qsoftware/synq`` (or ``pip install <path-to-synq>``
from a local copy). ``pip install synq`` installs an unrelated PyPI package.

SynQ returns the synthesized circuit as a gate list (numpy arrays) with the
gates ``rz``, ``ry``, ``p``, ``cx``, ``cz`` and ``h``. This module converts it into a
Qiskit circuit that follows the Qiskit (little-endian) qubit order and
reproduces the target exactly, including the global phase. ``qasm_to_circuit``
converts SynQ's OpenQASM 3 output, if that is preferred.
"""

import re
from math import isfinite, log2

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import CircuitInstruction
from qiskit.circuit.library import CXGate, CZGate, HGate, PhaseGate, RYGate, RZGate

try:  # Rust-native standard gates (Qiskit >= 1.3); private, so optional
    from qiskit._accelerate.circuit import StandardGate as _StandardGate

    _STANDARD = (
        _StandardGate.RZ, _StandardGate.RY, _StandardGate.Phase, _StandardGate.CX, _StandardGate.CZ,
        _StandardGate.H,
    )
    _FROM_STANDARD = CircuitInstruction.from_standard
except (ImportError, AttributeError):  # pragma: no cover - older/newer Qiskit
    _STANDARD = None
    _FROM_STANDARD = None

_INSTALL_HINT = (
    "SynQ is not installed. Install it with "
    "`pip install git+https://github.com/qsoftware/synq` "
    "(not `pip install synq`, which is an unrelated PyPI package)."
)

_UPDATE_HINT = (
    "The installed SynQ does not support method={method!r}. Update it with "
    "`pip install --upgrade git+https://github.com/qsoftware/synq`."
)

_RE_QUBITS = re.compile(r"^qubit\[(\d+)\]\s+(\w+)\s*;$")
_RE_ROTATION = re.compile(r"^(rz|ry|p)\(\s*([^)]+?)\s*\)\s+(\w+)\[(\d+)\]\s*;$")
_RE_CX = re.compile(r"^(cx|cz)\s+(\w+)\[(\d+)\]\s*,\s*(\w+)\[(\d+)\]\s*;$")
_RE_GPHASE = re.compile(r"^gphase\(\s*([^)]+?)\s*\)\s*;$")
_RE_H = re.compile(r"^h\s+(\w+)\[(\d+)\]\s*;$")


def _import_synq():
    try:
        import synq  # pylint: disable=import-outside-toplevel
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error
    return synq


def is_available():
    """Returns ``True`` if the SynQ Python module can be imported."""
    try:
        _import_synq()
    except ImportError:
        return False
    return True


def _parse_angle(text, line):
    value = float(text)
    if not isfinite(value):
        raise ValueError(f"SynQ produced a non-finite angle in line: {line!r}")
    return value


def qasm_to_circuit(qasm, name=None):
    """
    Converts an OpenQASM 3 program produced by SynQ into a ``QuantumCircuit``.

    Only the subset emitted by SynQ is accepted (one qubit register and the
    gates ``rz``, ``ry``, ``p``, ``cx``, ``cz``, ``h`` and ``gphase``); any other
    statement raises ``ValueError``.
    """
    circuit = None
    register = None
    global_phase = 0.0

    for number, raw in enumerate(qasm.splitlines(), start=1):
        line = raw.split("//", 1)[0].strip()
        if not line or line.startswith("OPENQASM") or line.startswith("include"):
            continue

        match = _RE_QUBITS.match(line)
        if match:
            if circuit is not None:
                raise ValueError("SynQ programs are expected to declare a single qubit register.")
            register = match.group(2)
            circuit = QuantumCircuit(int(match.group(1)), name=name)
            continue

        if circuit is None:
            raise ValueError(f"Gate before qubit declaration (line {number}): {raw!r}")

        match = _RE_ROTATION.match(line)
        if match:
            gate, angle, reg, qubit = match.groups()
            _check_register(reg, register, raw)
            getattr(circuit, gate)(_parse_angle(angle, raw), int(qubit))
            continue

        match = _RE_CX.match(line)
        if match:
            gate, reg_c, control, reg_t, target = match.groups()
            _check_register(reg_c, register, raw)
            _check_register(reg_t, register, raw)
            getattr(circuit, gate)(int(control), int(target))
            continue

        match = _RE_H.match(line)
        if match:
            reg, qubit = match.groups()
            _check_register(reg, register, raw)
            circuit.h(int(qubit))
            continue

        match = _RE_GPHASE.match(line)
        if match:
            global_phase += _parse_angle(match.group(1), raw)
            continue

        raise ValueError(f"Unsupported statement in SynQ output (line {number}): {raw!r}")

    if circuit is None:
        raise ValueError("SynQ output has no qubit declaration.")

    circuit.global_phase = global_phase
    return circuit


# Gate codes used by SynQ's gate lists.
_RZ, _RY, _P, _CX, _CZ, _H = 0, 1, 2, 3, 4, 5
_ROTATIONS = {_RZ: RZGate, _RY: RYGate, _P: PhaseGate}


def gates_to_circuit(num_qubits, gates, name=None):
    """
    Builds a ``QuantumCircuit`` from a SynQ gate list
    ``(types, qubits, angles, global_phase)``.

    Gate types: 0 ``rz``, 1 ``ry``, 2 ``p`` (``qubits[k, 0]`` is the target),
    3 ``cx``, 4 ``cz`` (``qubits[k]`` is ``(control, target)``) and 5 ``h``
    (``qubits[k, 0]`` is the target).
    """
    types, qubits, angles, global_phase = gates
    circuit = QuantumCircuit(num_qubits, name=name, global_phase=global_phase)
    types = np.asarray(types).tolist()
    qubits = np.asarray(qubits).tolist()
    angles = np.asarray(angles).tolist()
    # Fastest available method first; each fallback only depends on more public API.
    for append in (_append_standard, _append_fast, _append_public):
        try:
            append(circuit, types, qubits, angles)
            return circuit
        except (AttributeError, TypeError, NotImplementedError):  # pragma: no cover
            circuit = QuantumCircuit(num_qubits, name=name, global_phase=global_phase)
    raise RuntimeError("Could not build the SynQ circuit.")  # pragma: no cover


def _append_standard(circuit, types, qubits, angles):
    # CircuitInstruction.from_standard creates the instruction in Rust from a
    # StandardGate and its parameters, without building Python RZGate/RYGate/CXGate
    # objects (about 4-5x faster than _append_fast for large circuits).
    if _FROM_STANDARD is None:
        raise NotImplementedError
    bits = circuit.qubits
    append = circuit._data.append  # pylint: disable=protected-access
    standard, from_standard = _STANDARD, _FROM_STANDARD
    no_params = ()
    for gate, (q0, q1), angle in zip(types, qubits, angles):
        if gate < _CX:
            append(from_standard(standard[gate], (bits[q0],), (angle,)))
        elif gate == _H:
            append(from_standard(standard[gate], (bits[q0],), no_params))
        else:
            append(from_standard(standard[gate], (bits[q0], bits[q1]), no_params))


def _append_fast(circuit, types, qubits, angles):
    # Appending CircuitInstructions to the circuit data directly skips the argument
    # broadcasting and checks of QuantumCircuit.append, which are not needed for
    # this trusted input (about 2x faster for large circuits).
    bits = circuit.qubits
    append = circuit._data.append  # pylint: disable=protected-access
    cx_gate, cz_gate, h_gate = CXGate(), CZGate(), HGate()
    for gate, (q0, q1), angle in zip(types, qubits, angles):
        if gate == _CX:
            append(CircuitInstruction(cx_gate, (bits[q0], bits[q1])))
        elif gate == _CZ:
            append(CircuitInstruction(cz_gate, (bits[q0], bits[q1])))
        elif gate == _H:
            append(CircuitInstruction(h_gate, (bits[q0],)))
        else:
            append(CircuitInstruction(_ROTATIONS[gate](angle), (bits[q0],)))


def _append_public(circuit, types, qubits, angles):
    for gate, (q0, q1), angle in zip(types, qubits, angles):
        if gate == _CX:
            circuit.cx(q0, q1)
        elif gate == _CZ:
            circuit.cz(q0, q1)
        elif gate == _H:
            circuit.h(q0)
        else:
            circuit.append(_ROTATIONS[gate](angle), [q0])


def _check_register(found, expected, line):
    if found != expected:
        raise ValueError(f"Unknown register {found!r} in SynQ output: {line!r}")


def _num_qubits(length, what):
    n_qubits = log2(length) if length > 0 else 0.5
    if length < 2 or not float(n_qubits).is_integer():
        raise ValueError(f"The length of the {what} must be a power of 2 (>= 2).")
    return int(n_qubits)


def state_preparation(state):
    """
    Circuit that prepares ``state`` from ``|0...0>`` using SynQ
    (uniformly controlled RY/RZ rotations).

    Parameters
    ----------
    state: array_like of complex
        Normalized vector of length ``2^n``.

    Returns
    -------
    QuantumCircuit
    """
    state = np.asarray(state, dtype=complex).ravel()
    n_qubits = _num_qubits(len(state), "state vector")
    gates = _import_synq().state_preparation_gates(state)
    return gates_to_circuit(n_qubits, gates, name="synq_sp")


def unitary(gate, method="qsd"):
    """
    Circuit that implements the unitary matrix ``gate`` using SynQ.

    Parameters
    ----------
    gate: array_like of complex
        Unitary matrix of shape ``(2^n, 2^n)``.
    method: str
        ``"qsd"`` (default): cosine-sine / quantum Shannon decomposition with the
        optimizations A.1 and A.2 (20, 100, 444 CNOTs for 3, 4, 5 qubits).
        ``"zxz"``: Block-ZXZ decomposition with A.2 (19, 95, 423 CNOTs).

    Returns
    -------
    QuantumCircuit
    """
    gate = np.asarray(gate, dtype=complex)
    if gate.ndim != 2 or gate.shape[0] != gate.shape[1]:
        raise ValueError("The unitary must be a square matrix.")
    n_qubits = _num_qubits(gate.shape[0], "unitary matrix")
    if method not in ("qsd", "zxz"):
        raise ValueError(f"method must be 'qsd' or 'zxz', got {method!r}.")
    module = _import_synq()
    if method == "qsd":
        gates = module.unitary_gates(gate)  # also works with SynQ versions without `method`
    else:
        try:
            gates = module.unitary_gates(gate, method=method)
        except TypeError as error:  # SynQ older than the Block-ZXZ decomposition
            raise ImportError(_UPDATE_HINT.format(method=method)) from error
    return gates_to_circuit(n_qubits, gates, name=f"synq_{method}")


def diagonal(diag):
    """
    Circuit that implements the diagonal gate ``diag(diag)`` using SynQ.

    Parameters
    ----------
    diag: array_like of complex
        Diagonal entries (modulus 1), length ``2^n``.

    Returns
    -------
    QuantumCircuit
    """
    diag = np.asarray(diag, dtype=complex).ravel()
    n_qubits = _num_qubits(len(diag), "diagonal")
    gates = _import_synq().diagonal_gates(diag)
    return gates_to_circuit(n_qubits, gates, name="synq_diagonal")
